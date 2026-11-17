"""Verify array materialization against every SQL-produced itemstats cache row."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import duckdb
import numpy as np
import polars as pl

from src.common.feature_cache import cache_lock
from src.common.itemstats import FIELDS, KEY, feature_names, statistics_signature
from src.common.itemstats_lookup import LookupReader, export_lookup
from src.common.preparation import fingerprint, write_json
from src.common.resources import memory_snapshot
import src.common.itemstats_lookup as lookup_module


def benchmark(prepared, cache, output, *, memory_mb=1536):
    prepared, cache, output = map(lambda path: Path(path).resolve(), (prepared, cache, output))
    for source in (prepared, cache):
        if source == output or source in output.parents or output in source.parents:
            raise ValueError('benchmark output must be separate from source directories')
    if memory_mb < 128:
        raise ValueError('invalid benchmark memory limit')
    manifest_path = cache / 'manifest.json'
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    source_bytes = (prepared / 'manifest.json').read_bytes()
    source = json.loads(source_bytes)
    if not source.get('complete') or not manifest.get('complete') or manifest.get('feature_group') != 'itemstats':
        raise ValueError('complete prepared data and itemstats cache required')
    if manifest['identity']['prepared_manifest_sha256'] != hashlib.sha256(source_bytes).hexdigest():
        raise ValueError('prepared manifest differs from itemstats source')
    for relative, stamp in manifest['identity']['input_fingerprints'].items():
        if fingerprint(prepared / relative) != stamp:
            raise ValueError('itemstats source input changed')
    database = cache / '_statistics.duckdb'
    before = fingerprint(database)
    if before != manifest['statistics_fingerprint']:
        raise ValueError('itemstats statistics database changed')
    with cache_lock(output):
        result_path = output / 'run.json'
        result = {'complete': False, 'started_unix': time.time(), 'bitwise_equal_rows': 0,
                  'features': len(feature_names()), 'sample_materialization_times': [],
                  'prepared_manifest_sha256': hashlib.sha256(source_bytes).hexdigest(),
                  'cache_manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
                  'sql_producer_sha256': manifest['identity']['implementation_sha256'],
                  'statistics_value_signature': statistics_signature(),
                  'lookup_reader_sha256': hashlib.sha256(Path(lookup_module.__file__).read_bytes()).hexdigest(),
                  'benchmark_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'memory_mb': memory_mb}
        write_json(result_path, result)
        connection = duckdb.connect(str(database), read_only=True, config={
            'memory_limit': f'{memory_mb}MB', 'threads': '1', 'preserve_insertion_order': 'false',
            'temp_directory': str(output / '_spill')})
        reader = None
        try:
            started = time.monotonic()
            for entity in ('item', 'category'):
                export_lookup(connection, output, entity, FIELDS)
            result['lookup_export_seconds'] = round(time.monotonic() - started, 3)
            reader = LookupReader(output, FIELDS)
            for split, info in manifest['splits'].items():
                entries = {entry['source']: entry for entry in source['splits'][split]['parts']}
                for number, (name, part) in enumerate(info['parts'].items()):
                    path = prepared / entries[name]['candidate_path']
                    if fingerprint(path) != part['source'] or fingerprint(cache / part['path']) != part['fingerprint']:
                        raise ValueError('benchmark candidate or feature input changed')
                    frame = pl.read_parquet(path, columns=[*KEY, 'item_id', 'category_id'])
                    started = time.monotonic()
                    actual = reader.build_part(frame, feature_names())
                    array_seconds = time.monotonic() - started
                    expected = pl.read_parquet(cache / part['path'])
                    if not actual.select(*KEY, 'item_id').equals(expected.select(*KEY, 'item_id')):
                        raise ValueError('lookup candidate occurrence keys differ')
                    if not np.array_equal(actual.select(feature_names()).to_numpy().view(np.uint32),
                                          expected.select(feature_names()).to_numpy().view(np.uint32)):
                        raise ValueError('array lookup differs from SQL cache bits')
                    result['bitwise_equal_rows'] += frame.height
                    if number == 0:
                        connection.register('batch', frame.to_arrow())
                        query = ('SELECT b.split,b.feed_id,b.candidate_index,b.item_id,' + ','.join(feature_names())
                                 + ' FROM batch b LEFT JOIN item_lookup USING(item_id) LEFT JOIN category_lookup USING(category_id)'
                                 + ' ORDER BY b.feed_id,b.candidate_index')
                        started = time.monotonic()
                        sql = pl.from_arrow(connection.execute(query).fetch_arrow_table())
                        sql_seconds = time.monotonic() - started
                        if not actual.equals(sql):
                            raise ValueError('array lookup differs from live SQL join')
                        result['sample_materialization_times'].append({'split': split, 'rows': frame.height,
                            'array_seconds': round(array_seconds, 3), 'sql_seconds': round(sql_seconds, 3)})
                    if (number + 1) % 16 == 0:
                        print(split, number + 1, result['bitwise_equal_rows'], flush=True)
            reader.close()
            reader = None
            connection.close()
            connection = None
            if (fingerprint(database) != before or manifest_path.read_bytes() != manifest_bytes
                    or (prepared / 'manifest.json').read_bytes() != source_bytes):
                raise ValueError('benchmark source changed')
            result.update(complete=True, database_unchanged=True, memory=memory_snapshot(),
                          elapsed_seconds=round(time.time() - result['started_unix'], 3),
                          scope='all SQL cache rows compared bitwise. Single sample timings exclude cache IO and writes. Memory peak covers the whole benchmark process')
            write_json(result_path, result)
            return result
        except BaseException as error:
            result.update(complete=False, error=repr(error))
            write_json(result_path, result)
            raise
        finally:
            if reader is not None:
                reader.close()
            if connection is not None:
                connection.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, default=Path('data/prepared'))
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', dest='output', type=Path, required=True)
    parser.add_argument('--memory-mb', type=int, default=1536)
    result = benchmark(**vars(parser.parse_args()))
    print(json.dumps(result['sample_materialization_times'], indent=2))
