"""Category co-occurrence from donor feed histories with cohort exclusions."""
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from src.common.itemstats import training_sources
from src.common.preparation import fingerprint, flat_values, write_json, write_parquet_atomic
from src.common.resources import memory_snapshot

KEY = ['split', 'feed_id', 'candidate_index']
RECENT = (10, 50)
DOMAIN = 256


def feature_names():
    return [f'cc_{operation}_{recent}' for recent in RECENT for operation in ('sum', 'max')]


def history_presence(feeds):
    history = feeds['history__category_id']
    if history.dtype != pl.List(pl.Int8) or history.null_count():
        raise ValueError('category histories must be non-null Int8 lists')
    values = flat_values(history).astype(np.int16) + 128
    if not len(values):
        return np.zeros((feeds.height, 0)), np.array([], dtype=np.int16)
    categories, inverse = np.unique(values, return_inverse=True)
    rows = np.repeat(np.arange(feeds.height), history.list.len().to_numpy())
    present = np.zeros((feeds.height, len(categories)), dtype=np.float64)
    present[rows, inverse] = 1.
    return present, categories


def normalized_counts(counts):
    if counts.shape != (DOMAIN, DOMAIN) or counts.dtype != np.int64:
        raise ValueError('invalid category count matrix')
    support = counts.diagonal().astype(np.float64)
    if (counts < 0).any() or not np.array_equal(counts, counts.T):
        raise ValueError('invalid category co-occurrence counts')
    if (counts > np.minimum(support[:, None], support[None, :])).any():
        raise ValueError('pair frequency exceeds category support')
    denominator = np.sqrt(support[:, None] * support[None, :])
    return np.divide(counts, denominator, out=np.zeros_like(denominator), where=denominator > 0)


def build_feature_part(candidates, feeds, scores):
    if feeds['feed_id'].is_duplicated().any():
        raise ValueError('duplicate history feed')
    history_presence(feeds)
    frame = candidates.sort('feed_id', 'candidate_index')
    mapping = feeds.select('feed_id').with_row_index('_history_row')
    frame = frame.join(mapping, on='feed_id', how='left', validate='m:1')
    if (frame['_history_row'].null_count() or frame['category_id'].null_count()
            or frame['category_id'].dtype != pl.Int8):
        raise ValueError('missing history or invalid candidate category')
    lengths = feeds['history__category_id'].list.len().to_numpy().astype(np.int64)
    offsets = np.r_[0, lengths.cumsum()]
    values = flat_values(feeds['history__category_id']).astype(np.int16) + 128
    width = max(RECENT)
    present = np.arange(width)[None, :] < lengths[:, None]
    indices = np.zeros((feeds.height, width), dtype=np.int16)
    if len(values):
        positions = offsets[:-1, None] + np.arange(width)[None, :]
        indices = values[np.minimum(positions, len(values) - 1)]
    history_rows = frame['_history_row'].to_numpy()
    category = frame['category_id'].to_numpy().astype(np.int16) + 128
    output = {name: np.empty(frame.height, dtype=np.float32) for name in feature_names()}
    for start in range(0, frame.height, 16384):
        stop = min(frame.height, start + 16384)
        selected = history_rows[start:stop]
        similarity = scores[category[start:stop, None], indices[selected]]
        similarity = np.where(present[selected], similarity, 0.)
        for recent in RECENT:
            output[f'cc_sum_{recent}'][start:stop] = similarity[:, :recent].sum(axis=1)
            output[f'cc_max_{recent}'][start:stop] = similarity[:, :recent].max(axis=1)
    result = frame.select(*KEY, 'item_id').with_columns([
        pl.Series(name, values, dtype=pl.Float32) for name, values in output.items()])
    validate_features(result, candidates.height)
    return result


def validate_features(frame, expected_rows):
    schema = {'split': pl.String, 'feed_id': pl.Int32, 'candidate_index': pl.UInt32, 'item_id': pl.Int32,
              **{name: pl.Float32 for name in feature_names()}}
    if frame.height != expected_rows or frame.schema != schema or frame.select(KEY).is_duplicated().any():
        raise ValueError('invalid category co-occurrence schema, rows or keys')
    if frame.null_count().to_numpy().any():
        raise ValueError('null co-occurrence feature')
    for recent in RECENT:
        for operation, limit in (('sum', recent), ('max', 1)):
            values = frame[f'cc_{operation}_{recent}']
            if not values.is_finite().all() or (values < 0).any() or (values > limit + 1e-5).any():
                raise ValueError('category co-occurrence feature outside its domain')


def save_counts(path, counts, processed, identity_sha256):
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, counts=counts, processed=np.int64(processed),
                            identity_sha256=np.array(identity_sha256))
    os.replace(temporary, path)


def build_features(prepared, output, *, train_parts=8, resume=False):
    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    if prepared == output or prepared in output.parents or output in prepared.parents:
        raise ValueError('feature output and prepared input must be separate')
    source_bytes = (prepared / 'manifest.json').read_bytes()
    source = json.loads(source_bytes)
    if not source.get('complete'):
        raise ValueError('prepared data incomplete')
    selected = training_sources(source, train_parts)
    excluded = {entry['source'] for entry in selected}
    donors = [entry for entry in source['splits']['train']['parts'] if entry['source'] not in excluded]
    outputs = {'train': selected, **{split: source['splits'][split]['parts'] for split in ('val', 'test')}}
    inputs = {entry['feed_path']: fingerprint(prepared / entry['feed_path']) for entry in donors}
    for entries in outputs.values():
        for entry in entries:
            for field in ('candidate_path', 'feed_path'):
                inputs[entry[field]] = fingerprint(prepared / entry[field])
    identity = {
        'prepared_manifest_sha256': hashlib.sha256(source_bytes).hexdigest(),
        'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'excluded_train_sources': sorted(excluded), 'donor_sources': [entry['source'] for entry in donors],
        'input_fingerprints': inputs,
    }
    identity_sha256 = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    manifest_path = output / 'manifest.json'
    previous = {}
    if manifest_path.exists():
        if not resume:
            raise FileExistsError('output exists; use --resume')
        previous = json.loads(manifest_path.read_text())
        if previous.get('identity') != identity:
            raise ValueError('co-occurrence sources changed; choose a new output directory')
        if (previous.get('complete') and fingerprint(output / '_counts.npz')
                != previous.get('statistics_fingerprint')):
            raise ValueError('completed count checkpoint changed')
    elif (output / '_counts.npz').exists():
        raise ValueError('orphan count matrix without a manifest')
    manifest = {
        'format_version': 1, 'feature_group': 'category_cooc', 'complete': False, 'key': KEY,
        'features': feature_names(), 'identity': identity, 'excluded_train_sources': sorted(excluded),
        'history_policy': 'one presence per category per donor feed; query windows retain repeated events',
        'normalization': 'C(i,j) / sqrt(C(i,i)*C(j,j)); identical categories included; unseen categories zero',
        'target_policy': 'only donor histories; donor shards disjoint from training; labels unused',
        'splits': previous.get('splits', {}),
    }
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    manifest['stage'] = 'donor_counts'
    write_json(manifest_path, manifest)
    state_path = output / '_counts.npz'
    counts, processed = np.zeros((DOMAIN, DOMAIN), dtype=np.int64), 0
    if resume and state_path.exists():
        with np.load(state_path, allow_pickle=False) as state:
            if str(state['identity_sha256']) != identity_sha256:
                raise ValueError('count checkpoint identity differs')
            counts, processed = state['counts'], int(state['processed'])
        normalized_counts(counts)
        if not 0 <= processed <= len(donors):
            raise ValueError('invalid donor checkpoint progress')
    for number in range(processed, len(donors)):
        entry = donors[number]
        feeds = pl.read_parquet(prepared / entry['feed_path'], columns=['history__category_id'])
        presence, categories = history_presence(feeds)
        counts[np.ix_(categories, categories)] += (presence.T @ presence).astype(np.int64)
        save_counts(state_path, counts, number + 1, identity_sha256)
        if number == 0 or (number + 1) % 64 == 0 or number + 1 == len(donors):
            manifest['donors_processed'] = number + 1
            manifest['resources'] = memory_snapshot()
            write_json(manifest_path, manifest)
            print(f'category cooc donors: {number + 1}/{len(donors)}', flush=True)
    scores = normalized_counts(counts)
    manifest['statistics_fingerprint'] = fingerprint(state_path)
    for split, entries in outputs.items():
        old_parts = previous.get('splits', {}).get(split, {}).get('parts', {})
        def reusable_entry(entry):
            old = old_parts.get(entry['source'], {})
            path = output / split / entry['source']
            return (old.get('source') == fingerprint(prepared / entry['candidate_path'])
                    and old.get('feeds_source') == fingerprint(prepared / entry['feed_path'])
                    and path.exists() and old.get('fingerprint') == fingerprint(path)
                    and pq.read_metadata(path).num_rows == entry['candidates'])
        saved = {entry['source']: dict(old_parts[entry['source']]) for entry in entries if reusable_entry(entry)}
        for number, entry in enumerate(entries, 1):
            path = output / split / entry['source']
            candidate_stamp = fingerprint(prepared / entry['candidate_path'])
            feed_stamp = fingerprint(prepared / entry['feed_path'])
            reusable = entry['source'] in saved
            if not reusable:
                candidates = pl.read_parquet(prepared / entry['candidate_path'],
                                             columns=[*KEY, 'item_id', 'category_id'])
                feeds = pl.read_parquet(prepared / entry['feed_path'], columns=['feed_id', 'history__category_id'])
                features = build_feature_part(candidates, feeds, scores)
                if not features.select(*KEY, 'item_id').equals(
                        candidates.sort('feed_id', 'candidate_index').select(*KEY, 'item_id')):
                    raise ValueError('candidate keys changed')
                write_parquet_atomic(features, path)
            saved[entry['source']] = {
                'path': path.relative_to(output).as_posix(), 'rows': entry['candidates'],
                'source': candidate_stamp, 'feeds_source': feed_stamp, 'fingerprint': fingerprint(path),
            }
            manifest['splits'][split] = {'parts': saved, 'covered_parts': len(saved),
                                         'total_parts': len(source['splits'][split]['parts'])}
            manifest['stage'] = f'features_{split}'
            write_json(manifest_path, manifest)
            if number == len(entries):
                print(f'category cooc {split}: {number} parts complete', flush=True)
    for relative, stamp in inputs.items():
        if fingerprint(prepared / relative) != stamp:
            raise ValueError('input changed during co-occurrence generation')
    if (prepared / 'manifest.json').read_bytes() != source_bytes:
        raise ValueError('prepared manifest changed')
    manifest.update(complete=True, stage='complete', duration_seconds=round(time.monotonic() - started, 2),
                    resources=memory_snapshot(), donors_processed=len(donors))
    write_json(manifest_path, manifest)
    return manifest
