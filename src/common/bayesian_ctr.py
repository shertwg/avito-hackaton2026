"""Hierarchical item/category CTR over strictly past complete donor days."""
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path

import duckdb
import polars as pl

from src.common.feature_cache import KEY, build_cache, cache_lock, selected_parts
from src.common.item2item import checkpoint_table
from src.common.preparation import fingerprint, write_json

WINDOWS = ('all', '3d', '7d')


def feature_names():
    return [f'bc_{field}_{window}' for window in WINDOWS
            for field in ('item_ctr', 'category_ctr', 'item_category_delta', 'item_log_count')] + ['bc_donor_age_days', 'bc_has_past']


def domains():
    return {name: (-1, 1) if '_delta_' in name else (0, 64) if '_log_count_' in name
            else (0, 1e9) if name == 'bc_donor_age_days' else (0, 1) for name in feature_names()}


def aggregate_sql(query, daily, keys, last_day):
    selected = ','.join(f'q.{key}' for key in [*keys, 'day'])
    same = ' AND '.join(f'q.{key}=d.{key}' for key in keys) or 'true'
    cutoff = f'least(q.day-1,{last_day})'
    columns = []
    for window in WINDOWS:
        condition = '' if window == 'all' else f' FILTER (WHERE d.day>={cutoff}-{int(window[:-1])-1})'
        columns.extend(f'coalesce(sum(d.{field}){condition},0)::BIGINT {field}_{window}' for field in ('n', 'k'))
    if not keys:
        columns.append('max(d.day) last_day')
    return (f'SELECT {selected},{",".join(columns)} FROM ({query}) q LEFT JOIN {daily} d '
            f'ON {same} AND d.day<={cutoff} GROUP BY {selected}')


def ensure_lookups(prepared, output, donors, queries, days, signature, smoothing, memory_mb, threads):
    marker = output / '_stats.json'
    expected_files = {f'_lookup_{day}.parquet' for day in days}
    saved = json.loads(marker.read_text()) if marker.exists() else {}
    if saved.get('identity_sha256') != signature:
        saved = {'identity_sha256': signature, 'files': {}, 'complete': False}
    if (saved.get('complete') and set(saved['files']) == expected_files
            and all((output / name).exists() and fingerprint(output / name) == stamp for name, stamp in saved['files'].items())):
        return
    connection = duckdb.connect(str(output / '_stats.duckdb'), config={
        'memory_limit': f'{memory_mb}MB', 'threads': str(threads), 'preserve_insertion_order': 'false',
        'temp_directory': str(output / '_spill')})
    try:
        connection.from_parquet([str(prepared / p['candidate_path']) for p in queries]).create_view('queries', replace=True)
        connection.from_parquet([str(prepared / p['candidate_path']) for p in donors]).create_view('donors', replace=True)
        checkpoint_table(connection, 'query_keys', 'SELECT DISTINCT item_id,category_id,floor(timestamp/86400.)::INTEGER AS day FROM queries')
        checkpoint_table(connection, 'item_day',
            'SELECT d.item_id,d.category_id,floor(d.timestamp/86400.)::INTEGER AS day,count(*)::BIGINT n,sum(d.target)::BIGINT k '
            'FROM donors d SEMI JOIN (SELECT DISTINCT item_id,category_id FROM query_keys) q USING(item_id,category_id) '
            'WHERE d.item_id>0 GROUP BY d.item_id,d.category_id,day')
        checkpoint_table(connection, 'category_day',
            'SELECT category_id,floor(timestamp/86400.)::INTEGER AS day,count(*)::BIGINT n,sum(target)::BIGINT k '
            'FROM donors GROUP BY category_id,day')
        checkpoint_table(connection, 'global_day', 'SELECT day,sum(n)::BIGINT n,sum(k)::BIGINT k FROM category_day GROUP BY day')
        last_day = connection.execute('SELECT max(day) FROM global_day').fetchone()[0]
        last_day = int(last_day) if last_day is not None else min(days) - 1
        checkpoint_table(connection, 'category_sums', aggregate_sql('SELECT DISTINCT category_id,day FROM query_keys', 'category_day', ['category_id'], last_day))
        checkpoint_table(connection, 'global_sums', aggregate_sql('SELECT DISTINCT day FROM query_keys', 'global_day', [], last_day))
        expressions = []
        for window in WINDOWS:
            prior = f'(g.k_{window}::DOUBLE/greatest(g.n_{window},1))'
            category = f'((c.k_{window}+{smoothing}*{prior})/(c.n_{window}+{smoothing}))'
            item = f'((i.k_{window}+{smoothing}*{category})/(i.n_{window}+{smoothing}))'
            expressions.extend((f'{item}::FLOAT bc_item_ctr_{window}', f'{category}::FLOAT bc_category_ctr_{window}',
                f'({item}-{category})::FLOAT bc_item_category_delta_{window}', f'ln(1+i.n_{window})::FLOAT bc_item_log_count_{window}'))
        expressions.extend(('CASE WHEN g.n_all>0 THEN i.day-g.last_day ELSE 0 END::FLOAT bc_donor_age_days',
                            '(g.n_all>0)::INTEGER::FLOAT bc_has_past'))
        for day in days:
            name = f'_lookup_{day}.parquet'
            path = output / name
            if path.exists() and saved['files'].get(name) == fingerprint(path):
                continue
            item_query = aggregate_sql(f'SELECT * FROM query_keys WHERE day={int(day)}',
                                       'item_day', ['item_id', 'category_id'], last_day)
            relation = connection.sql(f'SELECT i.item_id,i.category_id,i.day,{",".join(expressions)} '
                f'FROM ({item_query}) i JOIN category_sums c USING(category_id,day) JOIN global_sums g USING(day) '
                'ORDER BY i.item_id,i.category_id')
            temporary = path.with_suffix('.parquet.tmp')
            relation.write_parquet(str(temporary), compression='zstd')
            temporary.replace(path)
            saved['files'][name] = fingerprint(path)
            write_json(marker, saved)
            print(f'bayesian CTR lookup day {day} ready', flush=True)
        impressions, clicks = connection.execute('SELECT coalesce(sum(n),0),coalesce(sum(k),0) FROM global_day').fetchone()
        saved.update(complete=True, donor_parts=len(donors), donor_impressions=int(impressions), donor_clicks=int(clicks), last_day=last_day)
        write_json(marker, saved)
    finally:
        connection.close()


def build_features(prepared, output, *, train_parts=8, smoothing=20., memory_mb=1536, threads=1, resume=False):
    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    if not math.isfinite(smoothing) or smoothing <= 0 or memory_mb < 128 or threads < 1:
        raise ValueError('invalid CTR smoothing or resource parameters')
    source = json.loads((prepared / 'manifest.json').read_text())
    if not source.get('complete'):
        raise ValueError('complete prepared data required')
    selected = selected_parts(source, train_parts)
    excluded = {p['source'] for p in selected}
    donors = [p for p in source['splits']['train']['parts'] if p['source'] not in excluded]
    if not donors:
        raise ValueError('Bayesian CTR requires separate donor shards')
    queries = [*selected, *source['splits']['val']['parts'], *source['splits']['test']['parts']]
    days = sorted(pl.scan_parquet([prepared / p['candidate_path'] for p in queries]).select(
        (pl.col('timestamp').cast(pl.Int64) // 86400).cast(pl.Int32).alias('day')).unique().collect()['day'].to_list())

    @contextmanager
    def prepare():
        identity = json.loads((output / 'manifest.json').read_text())['identity']
        signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        ensure_lookups(prepared, output, donors, queries, days, signature, smoothing, memory_mb, threads)
        yield None

    def build_part(split, entry, state):
        frame = pl.read_parquet(prepared / entry['candidate_path'], columns=[*KEY, 'item_id', 'category_id', 'timestamp'])
        frame = frame.with_columns((pl.col('timestamp').cast(pl.Int64) // 86400).cast(pl.Int32).alias('day'))
        chunks = []
        for day in frame['day'].unique().to_list():
            lookup = pl.read_parquet(output / f'_lookup_{day}.parquet')
            current = frame.filter(pl.col('day') == day)
            chunks.append(current.join(lookup, on=['item_id', 'category_id', 'day'], how='left', validate='m:1').select(
                *KEY, 'item_id', *feature_names()))
        return pl.concat(chunks).sort('feed_id', 'candidate_index')

    manifest = build_cache(prepared, output, train_parts=train_parts, group='bayesian_ctr', names=feature_names(), domains=domains(),
        producer=__file__, extra={'days': days, 'smoothing': smoothing, 'windows': list(WINDOWS),
            'aggregation': 'query-day batches',
            'cutoff_policy': 'strictly previous complete UTC days; windows anchored at min(query_day-1,last_donor_day)',
            'prior': 'global window rate -> category -> item; zero when no eligible donor observations'},
        excluded=excluded, prepare=prepare, build_part=build_part, resume=resume,
        artifacts=('_stats.json', *[f'_lookup_{day}.parquet' for day in days]))
    with cache_lock(output):
        current = json.loads((output / 'manifest.json').read_text())
        if current['identity'] != manifest['identity'] or not current.get('complete'):
            raise ValueError('CTR cache changed before target policy finalization')
        current['target_policy'] = 'past donor train target counts; query data supplies keys and timestamps'
        write_json(output / 'manifest.json', current)
    return current
