"""UTC calendar fields and age since first strictly earlier donor appearance."""
from contextlib import contextmanager
import json
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from src.common.feature_cache import KEY, build_cache, selected_parts

NAMES = ['time_hour_utc', 'time_weekday_utc', 'item_train_age_days', 'item_seen_before']


def feature_names():
    return list(NAMES)


def domains():
    return dict(zip(NAMES, ((0, 23), (0, 6), (0, None), (0, 1))))


def add_time_fields(frame):
    timestamps = frame['timestamp'].to_numpy().astype(np.int64)
    first_seen = frame['first_seen'].fill_null(0).to_numpy().astype(np.int64)
    known = frame['first_seen'].is_not_null().to_numpy() & (frame['item_id'].to_numpy() > 0) & (first_seen < timestamps)
    values = [timestamps // 3600 % 24, (timestamps // 86400 + 3) % 7,
              np.where(known, timestamps - first_seen, 0) / 86400., known]
    return frame.select(*KEY, 'item_id').with_columns([
        pl.Series(name, values, dtype=pl.Float32) for name, values in zip(NAMES, values)])


def build_features(prepared, output, *, train_parts=8, memory_mb=1536, resume=False, first_train_shards=False):
    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    if memory_mb < 128:
        raise ValueError('memory limit must be at least 128 MB')
    source = json.loads((prepared / 'manifest.json').read_text())
    selected = selected_parts(source, train_parts, first_train_shards=first_train_shards)
    excluded = {entry['source'] for entry in selected}
    donors = [entry for entry in source['splits']['train']['parts'] if entry['source'] not in excluded]
    if not donors:
        raise ValueError('item age requires separate donor shards')
    queries = [*selected, *source['splits']['val']['parts'], *source['splits']['test']['parts']]

    @contextmanager
    def aggregate():
        connection = duckdb.connect(str(output / '_first_seen.duckdb'), config={
            'memory_limit': f'{memory_mb}MB', 'threads': '1', 'preserve_insertion_order': 'false',
            'temp_directory': str(output / '_spill')})
        try:
            connection.from_parquet([str(prepared / entry['candidate_path']) for entry in donors]).create_view('donors', replace=True)
            connection.from_parquet([str(prepared / entry['candidate_path']) for entry in queries]).create_view('queries', replace=True)
            invalid = connection.execute('SELECT count(*) FROM donors WHERE item_id IS NULL OR timestamp IS NULL').fetchone()[0]
            if invalid:
                raise ValueError('null donor identifiers or timestamps')
            connection.execute('CREATE TABLE IF NOT EXISTS query_items AS SELECT DISTINCT item_id FROM queries WHERE item_id>0')
            connection.execute('CHECKPOINT')
            print('time features: query items ready', flush=True)
            connection.execute('CREATE TABLE IF NOT EXISTS first_seen AS '
                               'SELECT item_id,min(timestamp)::BIGINT first_seen FROM donors '
                               'SEMI JOIN query_items USING (item_id) GROUP BY item_id')
            connection.execute('CHECKPOINT')
            print('time features: first donor appearances ready', flush=True)
            yield connection
        finally:
            connection.close()

    def build_part(split, entry, connection):
        candidates = pl.read_parquet(prepared / entry['candidate_path'], columns=[*KEY, 'item_id', 'timestamp'])
        if candidates.null_count().to_numpy().any():
            raise ValueError('null query identifiers or timestamps')
        connection.register('batch', candidates.to_arrow())
        try:
            frame = pl.from_arrow(connection.execute(
                'SELECT b.*,f.first_seen FROM batch b LEFT JOIN first_seen f USING (item_id) '
                'ORDER BY b.feed_id,b.candidate_index').fetch_arrow_table())
        finally:
            connection.unregister('batch')
        return add_time_fields(frame)

    extra = {'donor_sources': [entry['source'] for entry in donors], 'timezone': 'UTC',
             'age_policy': 'first positive-ID donor candidate timestamp strictly before query; unknown age zero'}
    return build_cache(prepared, output, train_parts=train_parts, group='time', names=feature_names(), domains=domains(),
                       producer=__file__, extra=extra, build_part=build_part, resume=resume, excluded=excluded,
                       prepare=aggregate, artifacts=('_first_seen.duckdb',), first_train_shards=first_train_shards)
