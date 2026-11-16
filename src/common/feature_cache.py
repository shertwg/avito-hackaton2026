"""Shared occurrence-keyed cache handling for additional feature groups."""
from contextlib import contextmanager, nullcontext
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from src.common.preparation import fingerprint, write_json, write_parquet_atomic
from src.common.resources import memory_snapshot

KEY = ['split', 'feed_id', 'candidate_index']


def selected_parts(source, count, *, first_train_shards=False):
    parts = source['splits']['train']['parts']
    if not 1 <= count <= len(parts):
        raise ValueError('training shard count outside source range')
    if first_train_shards:
        return parts[:count]
    return [parts[int(index)] for index in np.linspace(0, len(parts) - 1, count, dtype=int)]


def validate_frame(frame, names, rows, domains):
    schema = {'split': pl.String, 'feed_id': pl.Int32, 'candidate_index': pl.UInt32, 'item_id': pl.Int32,
              **{name: pl.Float32 for name in names}}
    if frame.height != rows or frame.schema != schema or frame.select(KEY).is_duplicated().any():
        raise ValueError('invalid feature schema, row count or occurrence keys')
    if frame.null_count().to_numpy().any():
        raise ValueError('null feature value or key')
    for name in names:
        values = frame[name]
        lower, upper = domains[name]
        if (not values.is_finite().all() or (lower is not None and (values < lower - 1e-5).any())
                or (upper is not None and (values > upper + 1e-5).any())):
            raise ValueError(f'feature outside its domain: {name}')


@contextmanager
def cache_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    path = output / '_build.lock'
    with path.open('a+b') as stream:
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError('another writer is building this feature cache') from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def build_cache(prepared, output, *, train_parts, group, names, domains, producer,
                extra, build_part, resume=False, excluded=None, prepare=None,
                verify_extra=None, artifacts=(), first_train_shards=False):
    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    if prepared == output or prepared in output.parents or output in prepared.parents:
        raise ValueError('feature output must be separate from prepared data')
    source_bytes = (prepared / 'manifest.json').read_bytes()
    source = json.loads(source_bytes)
    if not source.get('complete'):
        raise ValueError('prepared dataset incomplete')
    requested = selected_parts(source, train_parts, first_train_shards=first_train_shards)
    identity = {
        'prepared_manifest_sha256': hashlib.sha256(source_bytes).hexdigest(),
        'implementation_sha256': hashlib.sha256(Path(producer).read_bytes()).hexdigest(),
        'cache_implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'extra': extra,
        'input_fingerprints': {
            entry[field]: fingerprint(prepared / entry[field])
            for info in source['splits'].values() for entry in info['parts']
            for field in ('candidate_path', 'feed_path')},
    }
    if excluded is not None:
        identity['excluded_train_sources'] = sorted(excluded)
        if set(excluded) != {entry['source'] for entry in requested}:
            raise ValueError('donor exclusions differ from requested training cohort')
    with cache_lock(output):
        manifest_path = output / 'manifest.json'
        previous = {}
        if manifest_path.exists():
            if not resume:
                raise FileExistsError('cache exists; use --resume')
            previous = json.loads(manifest_path.read_text(encoding='utf-8'))
            if previous.get('identity') != identity:
                raise ValueError('feature sources or implementation changed; choose a fresh output directory')
            if previous.get('complete'):
                for name, stamp in previous.get('artifacts', {}).items():
                    if fingerprint(output / name) != stamp:
                        raise ValueError('completed aggregate artifact changed')
        elif any(path.name != '_build.lock' for path in output.iterdir()):
            raise ValueError('nonempty cache directory has no source manifest')
        manifest = {
            'format_version': 1, 'feature_group': group, 'complete': False, 'key': KEY,
            'features': names, 'domains': domains, 'identity': identity,
            'splits': previous.get('splits', {}), 'target_policy': 'labels unused in feature computation',
        }
        if excluded is not None:
            manifest['excluded_train_sources'] = sorted(excluded)
        started = time.monotonic()
        write_json(manifest_path, manifest)
        context = prepare() if prepare else nullcontext(None)
        with context as state:
            for split, info in source['splits'].items():
                old_parts = previous.get('splits', {}).get(split, {}).get('parts', {})
                def reusable(entry):
                    old = old_parts.get(entry['source'], {})
                    path = output / split / entry['source']
                    return (old.get('source') == fingerprint(prepared / entry['candidate_path'])
                            and old.get('feeds_source') == fingerprint(prepared / entry['feed_path'])
                            and path.exists() and old.get('fingerprint') == fingerprint(path)
                            and pq.read_metadata(path).num_rows == entry['candidates'])
                entries = info['parts']
                if split == 'train':
                    required = {entry['source'] for entry in requested}
                    entries = [entry for entry in entries if entry['source'] in required
                               or (excluded is None and reusable(entry))]
                saved = {entry['source']: dict(old_parts[entry['source']]) for entry in entries if reusable(entry)}
                for number, entry in enumerate(entries, 1):
                    path = output / split / entry['source']
                    if entry['source'] not in saved:
                        phase_started = time.monotonic()
                        frame = build_part(split, entry, state)
                        validate_frame(frame, names, entry['candidates'], domains)
                        expected = pl.read_parquet(prepared / entry['candidate_path'], columns=[*KEY, 'item_id'])
                        if not frame.select(*KEY, 'item_id').equals(expected.sort('feed_id', 'candidate_index')):
                            raise ValueError('candidate occurrence keys changed')
                        write_parquet_atomic(frame, path)
                        saved[entry['source']] = {
                            'path': path.relative_to(output).as_posix(), 'rows': frame.height,
                            'source': fingerprint(prepared / entry['candidate_path']),
                            'feeds_source': fingerprint(prepared / entry['feed_path']),
                            'fingerprint': fingerprint(path),
                            'duration_seconds': round(time.monotonic() - phase_started, 3)}
                    manifest['splits'][split] = {'parts': saved, 'covered_parts': len(saved), 'total_parts': len(info['parts'])}
                    manifest['stage'] = f'features_{split}'
                    manifest['resources'] = memory_snapshot()
                    write_json(manifest_path, manifest)
                    if number == 1 or number % 16 == 0 or number == len(entries):
                        print(f'{group} {split}: {number}/{len(entries)} parts', flush=True)
        for relative, stamp in identity['input_fingerprints'].items():
            if fingerprint(prepared / relative) != stamp:
                raise ValueError('prepared input changed during generation')
        if ((prepared / 'manifest.json').read_bytes() != source_bytes
                or (verify_extra and verify_extra() != extra)):
            raise ValueError('source identity changed during generation')
        manifest.update(complete=True, stage='complete', duration_seconds=round(time.monotonic() - started, 2),
                        resources=memory_snapshot(), artifacts={name: fingerprint(output / name) for name in artifacts})
        write_json(manifest_path, manifest)
        return manifest
