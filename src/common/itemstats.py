"""Item/category statistics from train shards excluded from model training."""
import ast
import hashlib
import json
import math
from pathlib import Path
import time

import duckdb
import numpy as np
import polars as pl
import pyarrow.parquet as pq

from src.common.preparation import fingerprint, write_json, write_parquet_atomic
from src.common.resources import memory_snapshot
from src.common.train_selection import select_parts
from src.common import itemstats_lookup

KEY = ['split', 'feed_id', 'candidate_index']
EVENTS = tuple(range(1, 11))
FIELDS = ['impressions', 'clicks', 'ctr', 'expected_clicks', 'position_ratio',
          'history_count', *[f'history_evt{event}' for event in EVENTS]]


def feature_names():
    return [f'is_{entity}_{field}' for entity in ('item', 'category') for field in FIELDS]


def training_sources(source, train_parts, train_start=None):
    parts = source['splits']['train']['parts']
    if not 1 <= train_parts < len(parts):
        raise ValueError('itemstats requires at least one excluded donor train shard')
    return select_parts(source, train_parts, train_start)


def lookup_sql(entity, prior, smoothing):
    key = f'{entity}_id'
    expressions = {
        'impressions':'coalesce(e.impressions,0)', 'clicks':'coalesce(e.clicks,0)',
        'ctr':f'(coalesce(e.clicks,0)+{smoothing}*{prior})/(coalesce(e.impressions,0)+{smoothing})',
        'expected_clicks':'coalesce(e.expected_clicks,0)',
        'position_ratio':f'coalesce((coalesce(e.clicks,0)+{smoothing}*{prior})/nullif(coalesce(e.expected_clicks,0)+{smoothing}*{prior},0),1)',
        'history_count':'coalesce(h.history_count,0)',
        **{f'history_evt{event}':f'coalesce(h.history_evt{event},0)' for event in EVENTS},
    }
    columns = ', '.join(f'CAST({value} AS FLOAT) AS is_{entity}_{name}' for name,value in expressions.items())
    return (f'SELECT q.{key}, {columns} FROM query_{entity} q '
            f'LEFT JOIN {entity}_exposure e USING ({key}) LEFT JOIN {entity}_history h USING ({key})')


# Known SQL readers with the same aggregation and lookup formulas.
LEGACY_SQL_SIGNATURE = '69f0b30975ecb9b21f51734f456c4f749a2c55870d678dad8da8baa145427568'
LEGACY_IMPLEMENTATIONS = {
    'b1785668cf57123ff452c42cd62321f0cf0bd5e91a5dbea4e4b199d22bd13d26': LEGACY_SQL_SIGNATURE,
    'a0cba9b1c209649c95643e219539d17694cdfcb12408a894cf3fa86e9d4b0910': LEGACY_SQL_SIGNATURE,
    'f4ffbbbc8dc9f97ee652f0eb3974cbf08cc36a8fa202c623e1e8de0a82ddfcc0': LEGACY_SQL_SIGNATURE,
    '2622a241994047ba2871ce0aacb7527e6d3be2df142ace0b019279769882264a': LEGACY_SQL_SIGNATURE,
}


def statistics_signature():
    tree = ast.parse(Path(__file__).read_text(encoding='utf-8'))
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in {'KEY', 'EVENTS', 'FIELDS'} for target in node.targets):
            nodes.append(node)
        if isinstance(node, ast.FunctionDef) and node.name in {'feature_names', 'lookup_sql'}:
            nodes.append(node)
    builder = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'build_features')
    prefix = next(node for node in builder.body if isinstance(node, ast.Try)).body
    for node in prefix:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant)
                and target.slice.value == 'statistics_fingerprint' for target in node.targets):
            break
        nodes.append(node)
    return hashlib.sha256(ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False).encode()).hexdigest()


def complete_cache_valid(manifest, prepared, output, outputs):
    if not manifest.get('complete'):
        return False
    for relative, stamp in manifest['identity']['input_fingerprints'].items():
        if fingerprint(prepared / relative) != stamp:
            raise ValueError('itemstats input changed')
    if fingerprint(output / '_statistics.duckdb') != manifest['statistics_fingerprint']:
        raise ValueError('itemstats statistics database changed')
    for name, stamp in manifest.get('artifacts', {}).items():
        if not (output / name).exists() or fingerprint(output / name) != stamp:
            return False
    for split, entries in outputs.items():
        saved = manifest['splits'].get(split, {}).get('parts', {})
        if set(saved) != {entry['source'] for entry in entries}:
            return False
        for entry in entries:
            part = saved[entry['source']]
            path = output / part['path']
            if (not path.exists() or part['rows'] != entry['candidates']
                    or part['source'] != fingerprint(prepared / entry['candidate_path'])
                    or part['fingerprint'] != fingerprint(path)
                    or pq.read_metadata(path).num_rows != entry['candidates']):
                return False
    return True


def build_features(prepared, output, *, train_parts=8, smoothing=20., memory_mb=1024, threads=2, resume=False,
                   train_start=None):
    prepared,output = Path(prepared).resolve(),Path(output).resolve()
    if prepared == output or prepared in output.parents or output in prepared.parents:
        raise ValueError('itemstats output must be separate from prepared data')
    if not math.isfinite(smoothing) or smoothing <= 0 or memory_mb < 128 or threads < 1:
        raise ValueError('invalid smoothing or resource parameters')
    source_bytes = (prepared/'manifest.json').read_bytes()
    source = json.loads(source_bytes)
    if not source.get('complete'):
        raise ValueError('prepared dataset incomplete')
    selected = training_sources(source,train_parts,train_start)
    excluded = {entry['source'] for entry in selected}
    donors = [entry for entry in source['splits']['train']['parts'] if entry['source'] not in excluded]
    outputs = {'train':selected, **{split:source['splits'][split]['parts'] for split in ('val','test')}}
    inputs = {}
    for entry in [*donors, *[part for group in outputs.values() for part in group]]:
        for field in ('candidate_path','feed_path'):
            relative = entry[field]
            inputs[relative] = fingerprint(prepared/relative)
    identity = {'prepared_manifest_sha256':hashlib.sha256(source_bytes).hexdigest(),
                'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'excluded_train_sources':sorted(excluded),'donor_sources':[entry['source'] for entry in donors],
                'smoothing':smoothing,'input_fingerprints':inputs}
    if train_start is not None:
        identity['train_start'] = train_start
    identity['statistics_value_signature'] = statistics_signature()
    identity['lookup_implementation_sha256'] = hashlib.sha256(Path(itemstats_lookup.__file__).read_bytes()).hexdigest()
    previous = {}
    manifest_path = output/'manifest.json'
    if not manifest_path.exists() and (output/'_statistics.duckdb').exists():
        raise ValueError('statistics database has no source manifest; choose a new output directory')
    if manifest_path.exists():
        if not resume:
            raise FileExistsError('itemstats output exists; use --resume')
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        old_identity = previous.get('identity', {})
        legacy_identity = {key: value for key, value in identity.items() if key not in ('statistics_value_signature', 'lookup_implementation_sha256')}
        legacy_identity['implementation_sha256'] = old_identity.get('implementation_sha256')
        compatible_legacy = (LEGACY_IMPLEMENTATIONS.get(old_identity.get('implementation_sha256')) == identity['statistics_value_signature']
                             and old_identity == legacy_identity)
        if old_identity != identity and not compatible_legacy:
            raise ValueError('itemstats sources or configuration changed; choose a new output directory')
        if previous.get('features') != feature_names() or previous.get('key') != KEY:
            raise ValueError('itemstats feature contract changed')
        if complete_cache_valid(previous, prepared, output, outputs):
            return previous
    manifest = {'format_version':1,'feature_group':'itemstats','complete':False,'key':KEY,
                'features':feature_names(),'identity':identity,'excluded_train_sources':sorted(excluded),
                'target_policy':'train donor shards disjoint from model training; val/test targets unused',
                'history_policy':'event occurrences in donor histories, including repeated events',
                'resources_config':{'memory_mb':memory_mb,'threads':threads},
                'splits':previous.get('splits',{}),'tables':previous.get('tables',{})}
    previous_producers = previous.get('compatible_producer_implementations', [])
    if previous:
        previous_producers = [*previous_producers, previous['identity']['implementation_sha256']]
    manifest['compatible_producer_implementations'] = sorted(set([*previous_producers, identity['implementation_sha256']]))
    started = time.monotonic()
    output.mkdir(parents=True,exist_ok=True)
    connection = duckdb.connect(str(output/'_statistics.duckdb'),config={
        'memory_limit':f'{memory_mb}MB','threads':str(threads),
        'temp_directory':str(output/'_spill'),'preserve_insertion_order':'false'})
    reader = None
    try:
        write_json(manifest_path,manifest)
        connection.from_parquet([str(prepared/entry['candidate_path']) for entry in donors]).create_view('donors',replace=True)
        connection.from_parquet([str(prepared/entry['feed_path']) for entry in donors]).create_view('donor_feeds',replace=True)
        connection.from_parquet([str(prepared/entry['candidate_path']) for group in outputs.values() for entry in group]).create_view('queries',replace=True)
        connection.execute('CREATE OR REPLACE VIEW history_events AS SELECT unnest(history__item_id) item_id, '
                           'unnest(history__category_id) category_id, unnest(history__event_type) event_type FROM donor_feeds')

        def materialize(name, query):
            phase_started = time.monotonic()
            connection.execute(f'CREATE TABLE IF NOT EXISTS {name} AS {query}')
            connection.execute('CHECKPOINT')
            manifest['tables'][name] = {'rows':connection.execute(f'SELECT count(*) FROM {name}').fetchone()[0],
                                        'last_call_seconds':round(time.monotonic()-phase_started,3)}
            manifest['resources'] = memory_snapshot()
            manifest['stage'] = name
            write_json(manifest_path,manifest)
            print(f'itemstats {name}: {manifest["tables"][name]["rows"]:,} rows; {time.monotonic()-started:.0f}s',flush=True)

        materialize('query_item','SELECT DISTINCT item_id FROM queries')
        materialize('query_category','SELECT DISTINCT category_id FROM queries')
        invalid = connection.execute('SELECT count(*) FROM donors WHERE target IS NULL OR target NOT IN (0,1) OR pos IS NULL OR pos < 0').fetchone()[0]
        if invalid:
            raise ValueError('invalid donor labels or positions')
        invalid = connection.execute('SELECT count(*) FROM donor_feeds WHERE '
                                     'len(history__item_id) != len(history__category_id) OR '
                                     'len(history__item_id) != len(history__event_type)').fetchone()[0]
        if invalid:
            raise ValueError('unaligned donor history arrays')
        materialize('position_ctr','SELECT pos, count(*) impressions, sum(target)::BIGINT clicks, avg(target::DOUBLE) rate FROM donors GROUP BY pos')
        impressions,clicks = connection.execute('SELECT sum(impressions),sum(clicks) FROM position_ctr').fetchone()
        if not impressions:
            raise ValueError('empty donor candidate data')
        prior = float(clicks/impressions)
        manifest['donor_summary'] = {'parts':len(donors),'impressions':int(impressions),'clicks':int(clicks),'prior_ctr':prior,
            'position_ctr':[dict(zip(('pos','impressions','clicks','rate'),row)) for row in connection.execute('SELECT * FROM position_ctr ORDER BY pos').fetchall()]}
        event_counts = ', '.join(f'count(*) FILTER (WHERE event_type={event}) AS history_evt{event}' for event in EVENTS)
        for entity in ('item','category'):
            key = f'{entity}_id'
            subset = f'SEMI JOIN query_item USING ({key})' if entity == 'item' else ''
            materialize(f'{entity}_exposure',f'SELECT {key},count(*) impressions,sum(target)::BIGINT clicks,'
                        f'sum(p.rate) expected_clicks FROM donors JOIN position_ctr p USING (pos) {subset} GROUP BY {key}')
            materialize(f'{entity}_history',f'SELECT {key},count(*) history_count,{event_counts} '
                        f'FROM history_events {subset} GROUP BY {key}')
            materialize(f'{entity}_lookup',lookup_sql(entity,prior,smoothing))
        manifest['statistics_fingerprint'] = fingerprint(output/'_statistics.duckdb')
        saved_artifacts = previous.get('artifacts', {})
        manifest['artifacts'] = {}
        for entity in ('item', 'category'):
            paths = itemstats_lookup.lookup_paths(output, entity)
            if not all(path.exists() and saved_artifacts.get(path.name) == fingerprint(path) for path in paths):
                itemstats_lookup.export_lookup(connection, output, entity, FIELDS)
            manifest['artifacts'].update({path.name: fingerprint(path) for path in paths})
            write_json(manifest_path, manifest)
        reader = itemstats_lookup.LookupReader(output, FIELDS)
        for split,entries in outputs.items():
            previous_parts = previous.get('splits',{}).get(split,{}).get('parts',{})
            def reusable(entry):
                old = previous_parts.get(entry['source'],{})
                path = output/split/entry['source']
                return (old.get('source') == fingerprint(prepared/entry['candidate_path'])
                        and path.exists() and old.get('fingerprint') == fingerprint(path)
                        and pq.read_metadata(path).num_rows == entry['candidates'])
            saved = {entry['source']:dict(previous_parts[entry['source']]) for entry in entries if reusable(entry)}
            for number,entry in enumerate(entries,1):
                part_started = time.monotonic()
                path = output/split/entry['source']
                stamp = fingerprint(prepared/entry['candidate_path'])
                old = previous_parts.get(entry['source'],{})
                reuse = reusable(entry)
                if not reuse:
                    candidates = pl.read_parquet(prepared/entry['candidate_path'],columns=[*KEY,'item_id','category_id'])
                    features = reader.build_part(candidates, feature_names())
                    validate_features(features,entry['candidates'])
                    if not features.select(*KEY,'item_id').equals(candidates.sort('feed_id','candidate_index').select(*KEY,'item_id')):
                        raise ValueError('itemstats candidate keys changed')
                    write_parquet_atomic(features,path)
                producer = identity['implementation_sha256']
                if reuse:
                    producer = old.get('producer_implementation_sha256', previous.get('identity', {}).get('implementation_sha256'))
                saved[entry['source']] = {'path':path.relative_to(output).as_posix(),'rows':entry['candidates'],
                    'source':stamp,'fingerprint':fingerprint(path),
                    'duration_seconds':old.get('duration_seconds',0.) if reuse else round(time.monotonic()-part_started,3),
                    'producer_implementation_sha256':producer}
                manifest['splits'][split] = {'parts':saved,'covered_parts':len(saved),
                                             'total_parts':len(source['splits'][split]['parts'])}
                manifest['stage'] = f'features_{split}'
                manifest['resources'] = memory_snapshot()
                write_json(manifest_path,manifest)
                if not reuse or number == len(entries):
                    print(f'itemstats {split}: {number}/{len(entries)} shards; {time.monotonic()-started:.0f}s',flush=True)
        for relative,stamp in inputs.items():
            if fingerprint(prepared/relative) != stamp:
                raise ValueError('source changed during itemstats generation')
        manifest.update(complete=True,stage='complete',all_train_parts=False,
                        duration_seconds=round(time.monotonic()-started,2),resources=memory_snapshot())
        write_json(manifest_path,manifest)
        return manifest
    finally:
        if reader is not None:
            reader.close()
        connection.close()


def validate_features(frame, expected_rows):
    schema = {'split':pl.String,'feed_id':pl.Int32,'candidate_index':pl.UInt32,'item_id':pl.Int32,
              **{name:pl.Float32 for name in feature_names()}}
    if frame.height != expected_rows or frame.schema != schema or frame.select(KEY).is_duplicated().any():
        raise ValueError('invalid itemstats schema, count or candidate keys')
    if frame.null_count().to_numpy().any():
        raise ValueError('null itemstats feature')
    for name in feature_names():
        if not frame[name].is_finite().all() or (frame[name] < 0).any():
            raise ValueError(f'invalid itemstats feature: {name}')
    for entity in ('item','category'):
        if (frame[f'is_{entity}_ctr'] > 1).any():
            raise ValueError('itemstats CTR exceeds one')
