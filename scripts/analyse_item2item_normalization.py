"""Explore fixed asymmetric co-history normalizations on pseudo-Public feeds."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import polars as pl

from src.common.cv import pseudo_split
from src.common.item2item import ItemHistoryIndex
from src.common.metric import per_feed_ndcg
from src.common.preparation import fingerprint, write_json
from src.common.resources import memory_snapshot

ALPHAS = (0., .25, .5, .75, 1.)


def feature_key(alpha, recency, operation, window):
    if recency and operation == 'sum':
        operation = 'mean'
    return f'a{round(alpha * 100):03d}_{"recency" if recency else "plain"}_{operation}_{window}'


def analyse(prepared, parent, weighted, output, *, sample_feeds=512, seed=42):
    if sample_feeds < 1:
        raise ValueError('sample feed count must be positive')
    prepared, parent, weighted, output = map(lambda path: Path(path).resolve(), (prepared, parent, weighted, output))
    started = time.monotonic()
    source = json.loads((prepared / 'manifest.json').read_text())
    base = json.loads((parent / 'manifest.json').read_text())
    extra = json.loads((weighted / 'manifest.json').read_text())
    if not all(m.get('complete') for m in (source, base, extra)):
        raise ValueError('complete data and feature caches required')
    source_sha = hashlib.sha256((prepared / 'manifest.json').read_bytes()).hexdigest()
    if any(m['identity']['prepared_manifest_sha256'] != source_sha for m in (base, extra)):
        raise ValueError('prepared manifest differs from feature inputs')
    dependency = extra['identity']['extra']
    if dependency['parent_identity'] != base['identity'] or dependency['parent_artifacts'] != base['artifacts']:
        raise ValueError('weighted and unweighted indices differ')
    before = {str(path): fingerprint(path) for path in (prepared / 'manifest.json', parent / 'manifest.json', weighted / 'manifest.json')}
    for root, manifest in ((parent, base), (weighted, extra)):
        for relative, stamp in manifest['identity']['input_fingerprints'].items():
            if fingerprint(prepared / relative) != stamp:
                raise ValueError('feature source changed')
            before[str(prepared / relative)] = stamp
        for name, stamp in manifest['artifacts'].items():
            if fingerprint(root / name) != stamp:
                raise ValueError('co-history index changed')
    metadata = pl.concat([pl.read_parquet(prepared / entry['feed_path'], columns=['feed_id', 'timestamp'])
                          for entry in source['splits']['val']['parts']])
    public, _ = pseudo_split(metadata['feed_id'].to_numpy(), metadata['timestamp'].to_numpy())
    selected = np.sort(np.random.default_rng(seed).choice(public, min(sample_feeds, len(public)), replace=False))
    candidates, references, histories = [], [], {}
    columns = ['feed_id', 'candidate_index', 'item_id']
    raw_names = [f'i2i_{operation}_{window}' for window in (10, 50) for operation in ('sum', 'max')]
    weighted_names = [f'wi2i_all_{operation}_{window}' for window in (10, 50) for operation in ('mean', 'max')]
    for entry in source['splits']['val']['parts']:
        path = prepared / entry['candidate_path']
        before[str(path)] = fingerprint(path)
        frame = pl.read_parquet(path, columns=[*columns, 'target', 'f1']).filter(pl.col('feed_id').is_in(selected))
        if frame.is_empty():
            continue
        candidates.append(frame)
        path = prepared / entry['feed_path']
        before[str(path)] = fingerprint(path)
        feeds = pl.read_parquet(path, columns=['feed_id', 'history__item_id']).filter(pl.col('feed_id').is_in(selected))
        histories.update(feeds.iter_rows())
        chunks = []
        for root, manifest, names in ((parent, base, raw_names), (weighted, extra, weighted_names)):
            part = manifest['splits']['val']['parts'][entry['source']]
            path = root / part['path']
            if fingerprint(path) != part['fingerprint'] or part['source'] != before[str(prepared / entry['candidate_path'])]:
                raise ValueError('candidate or feature cache fingerprint differs')
            before[str(path)] = part['fingerprint']
            chunks.append(pl.read_parquet(path, columns=[*columns, *names]).filter(pl.col('feed_id').is_in(selected)))
        references.append(chunks[0].join(chunks[1], on=columns, validate='1:1'))
    frame = pl.concat(candidates).join(pl.concat(references), on=columns, how='left', validate='1:1').sort('feed_id', 'candidate_index')
    if frame.null_count().to_numpy().any() or frame['feed_id'].n_unique() != len(selected):
        raise ValueError('diagnostic sample coverage differs')
    definitions = [(alpha, recency, operation, window) for alpha in ALPHAS for recency in (False, True)
                   for window in (10, 50) for operation in ('sum', 'max')]
    scores = {feature_key(*definition): np.zeros(frame.height, dtype=np.float32) for definition in definitions}
    ids, items = frame['feed_id'].to_numpy(), frame['item_id'].to_numpy()
    bounds = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]
    with ItemHistoryIndex(parent) as index:
        for number, (first, stop) in enumerate(zip(bounds[:-1], bounds[1:]), 1):
            history = histories[int(ids[first])]
            if history is None or None in history:
                raise ValueError('invalid diagnostic history')
            history = np.asarray(history[:50], dtype=np.int32)
            if not len(history):
                continue
            history_popularity = np.array([len(values) for values in index.posting_lists(history)], dtype=np.float64)
            decay = np.exp2(-np.arange(len(history), dtype=np.float64) / 10.)
            for start in range(first, stop, 64):
                end = min(stop, start + 64)
                similarity = index.similarities(items[start:end], history)
                candidate_popularity = np.array([len(values) for values in index.posting_lists(items[start:end])], dtype=np.float64)
                ratio = np.maximum(history_popularity[None, :], 1.) / np.maximum(candidate_popularity[:, None], 1.)
                for alpha in ALPHAS:
                    normalized = np.clip(similarity * np.power(ratio, alpha - .5), 0., 1.)
                    for recency in (False, True):
                        for window in (10, 50):
                            weights = decay[:window] if recency else np.ones(min(window, len(history)))
                            values = normalized[:, :window] * weights[None, :]
                            total = values.sum(axis=1)
                            if recency:
                                total /= weights.sum()
                            scores[feature_key(alpha, recency, 'sum', window)][start:end] = total
                            scores[feature_key(alpha, recency, 'max', window)][start:end] = values.max(axis=1)
            if number % 64 == 0:
                print('Normalized feeds', number, '/', len(selected), flush=True)
    for window in (10, 50):
        for operation in ('sum', 'max'):
            if not np.array_equal(scores[feature_key(.5, False, operation, window)], frame[f'i2i_{operation}_{window}'].to_numpy()):
                raise ValueError('symmetric scores differ from original item2item bits')
            name = f'wi2i_all_{"mean" if operation == "sum" else "max"}_{window}'
            if not np.array_equal(scores[feature_key(.5, True, operation, window)], frame[name].to_numpy()):
                raise ValueError('recency scores differ from original weighted item2item bits')
    variants = []
    for definition in definitions:
        name = feature_key(*definition)
        _, values = per_feed_ndcg(ids, frame['target'].to_numpy(), scores[name], cutoff=30)
        operation = 'mean' if definition[1] and definition[2] == 'sum' else definition[2]
        variants.append({'name': name, 'alpha': definition[0], 'recency': definition[1], 'operation': operation,
                         'window': definition[3], 'ndcg30': float(values.mean())})
    _, baseline = per_feed_ndcg(ids, frame['target'].to_numpy(), frame['f1'].to_numpy(), cutoff=30)
    if any(fingerprint(Path(path)) != stamp for path, stamp in before.items()):
        raise ValueError('normalization inputs changed')
    result = {'complete': True, 'scope': 'univariate diagnostics on a fixed pseudo-Public sample',
              'normalization': 'Cij / (candidate_popularity^alpha * history_popularity^(1-alpha))',
              'recency': 'half-life 10 history positions, weighted mean or weighted maximum',
              'sample_feeds': len(selected), 'sample_rows': frame.height, 'seed': seed,
              'alphas': list(ALPHAS), 'prepared_manifest_sha256': source_sha,
              'parent_manifest_sha256': hashlib.sha256((parent / 'manifest.json').read_bytes()).hexdigest(),
              'weighted_manifest_sha256': hashlib.sha256((weighted / 'manifest.json').read_bytes()).hexdigest(),
              'sample_feed_ids_sha256': hashlib.sha256(selected.astype(np.int32).tobytes()).hexdigest(),
              'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'input_fingerprints_sha256': hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest(),
              'reference_bits_verified': raw_names + weighted_names, 'f1_ndcg30': float(baseline.mean()),
              'variants': sorted(variants, key=lambda value: -value['ndcg30']),
              'duration_seconds': round(time.monotonic() - started, 3), 'memory': memory_snapshot()}
    write_json(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, default=Path('data/prepared'))
    parser.add_argument('--parent', type=Path, default=Path('data/features/item2item-008'))
    parser.add_argument('--weighted', type=Path, default=Path('data/features/weighted-item2item-008'))
    parser.add_argument('--out', dest='output', type=Path, required=True)
    parser.add_argument('--sample-feeds', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    report = analyse(**vars(parser.parse_args()))
    print(json.dumps(report['variants'][:8], indent=2))
