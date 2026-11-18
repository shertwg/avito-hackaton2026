"""Estimate an embedding direction for clicked items using labeled train feeds.

Validation and coefficient selection use the two temporal halves of pseudo-public.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from scripts.run_joint_oof import digest
from src.common.cv import pseudo_split
from src.common.metric import per_feed_ndcg
from src.common.preparation import write_json
from src.common.scores import feed_zscore
from src.common.similarity import prefetch_vectors, unit
from src.common.train_selection import select_parts


KEY = ['feed_id', 'item_id']
INDEX = 'item_id__embedding_idx'
WEIGHTS = [0., .05, .15, .3, .6]


def read_batches(raw, prepared, split, entries, selected=None):
    # Bound the size of the prefetched int8 vector table independently of the total shard count.
    batch_size = 4 if selected is None else len(entries)
    for first in range(0, len(entries), batch_size):
        parts = []
        for entry in entries[first:first + batch_size]:
            keys = pl.read_parquet(prepared / entry['candidate_path'], columns=[*KEY, INDEX, 'target'])
            feeds = pl.read_parquet(prepared / entry['feed_path'], columns=['feed_id', 'timestamp', 'user_embedding'])
            if selected is not None:
                keys = keys.filter(pl.col('feed_id').is_in(selected))
                feeds = feeds.filter(pl.col('feed_id').is_in(selected))
            if keys.height:
                parts.append((keys.sort('feed_id'), feeds.sort('feed_id')))
        if not parts:
            continue
        rows = np.unique(np.concatenate([keys[INDEX].to_numpy() for keys, _ in parts]))
        rows = rows[rows >= 0]
        vectors = prefetch_vectors(raw, split, rows)
        for keys, feeds in parts:
            ids, starts, counts = np.unique(keys['feed_id'].to_numpy(), return_index=True, return_counts=True)
            if not np.array_equal(ids, feeds['feed_id'].to_numpy()):
                raise ValueError('feed and candidate IDs differ')
            users = unit(np.asarray(feeds['user_embedding'].to_list(), dtype=np.float32))
            for start in range(0, len(ids), 128):
                stop = min(start + 128, len(ids))
                low, high = starts[start], starts[stop] if stop < len(ids) else keys.height
                frame = keys.slice(int(low), int(high - low))
                indices = frame[INDEX].to_numpy()
                valid = indices >= 0
                items = np.zeros((len(indices), 128), dtype=np.float32)
                items[valid] = unit(vectors[np.searchsorted(rows, indices[valid])])
                yield (frame, users[start:stop], items, valid, starts[start:stop] - low,
                       counts[start:stop], feeds['timestamp'].slice(start, stop-start).to_numpy())


def train(raw, prepared, entries):
    observed = set()
    latest = None
    all_users, all_directions = [], []
    for frame, users, items, valid, starts, _, times in read_batches(raw, prepared, 'train', entries):
        ids = frame['feed_id'].to_numpy()[starts]
        if observed.intersection(ids):
            raise ValueError('train feeds repeated across shards')
        observed.update(ids)
        latest = int(max(times.max(), latest)) if latest is not None else int(times.max())
        positive = (frame['target'].to_numpy() == 1) & valid
        negative = ~positive & valid
        n_pos = np.add.reduceat(positive, starts, dtype=np.int64)
        n_neg = np.add.reduceat(negative, starts, dtype=np.int64)
        eligible = (n_pos > 0) & (n_neg > 0)
        pos_sum = np.add.reduceat(items * positive[:, None], starts, axis=0)
        neg_sum = np.add.reduceat(items * negative[:, None], starts, axis=0)
        directions = pos_sum / np.maximum(n_pos[:, None], 1) - neg_sum / np.maximum(n_neg[:, None], 1)
        all_users.append(users[eligible])
        all_directions.append(directions[eligible].astype(np.float32))
    return np.concatenate(all_users), np.concatenate(all_directions), observed, latest


def fit_ridge(users, directions):
    x = np.column_stack([users, np.ones(len(users))])
    cov = x.T @ x
    penalty = np.full(129, .1 * cov.trace() / 129)
    penalty[-1] = .0001
    return np.linalg.solve(cov + np.diag(penalty), x.T @ directions)


def predict(raw, prepared, entries, public, users, directions, ridge):
    k = min(64, len(users))
    scores = []
    for keys, query, items, valid, starts, counts, _ in read_batches(raw, prepared, 'val', entries, public):
        linear = query @ ridge[:-1] + ridge[-1]
        # Equal weight for the 64 nearest train users in cosine distance.
        affinities = query @ users.T
        neighbor_idx = np.argpartition(affinities, -k, axis=1)[:, -k:]
        local = directions[neighbor_idx].mean(axis=1)
        # A missing user vector has no nearest neighbors.
        local[np.linalg.norm(query, axis=1) == 0] = directions.mean(axis=0)
        frame = keys.select(*KEY, 'target')
        for name, direction in [('ridge', linear), ('neighbors', local)]:
            values = np.einsum('ij,ij->i', np.repeat(direction, counts, axis=0), items)
            n_valid = np.add.reduceat(valid, starts, dtype=np.int64)
            mean = np.add.reduceat(values, starts) / np.maximum(n_valid, 1)
            frame = frame.with_columns(pl.Series(name, np.where(valid, values, np.repeat(mean, counts))))
        scores.append(frame)
    return pl.concat(scores)


def evaluate(frame, early, late):
    feed, target = frame['feed_id'].to_numpy(), frame['target'].to_numpy()
    raw = frame['score'].to_numpy()
    base = feed_zscore(feed, raw)
    out = {}
    for name in ('ridge', 'neighbors'):
        semantics = feed_zscore(feed, frame[name].to_numpy())

        def metric(feeds, weight):
            mask = np.isin(feed, feeds)
            values = raw[mask] if weight == 0 else base[mask] + weight * semantics[mask]
            return per_feed_ndcg(feed[mask], target[mask], values, cutoff=30)[1]

        grid = [float(metric(early, w).mean()) for w in WEIGHTS]
        weight = WEIGHTS[int(np.argmax(grid))]
        before, after = metric(late, 0), metric(late, weight)
        delta = after - before
        rng = np.random.default_rng(42)
        draws = [rng.choice(delta, len(delta)).mean() for _ in range(1000)]
        out[name] = dict(selection_ndcg30=grid, weight=weight, late_base=float(before.mean()),
                         late_adjusted=float(after.mean()), delta=float(delta.mean()),
                         paired_ci95=np.quantile(draws, [.025, .975]).tolist())
    return out


def probe(raw, prepared, model, count):
    manifest = json.loads((prepared / 'manifest.json').read_bytes())
    report = json.loads((model / 'run.json').read_bytes())
    score_path = model / 'val_pred.parquet'
    if (not report['complete'] or digest(prepared / 'manifest.json') != report['identity']['prepared_manifest_sha256']
            or digest(score_path) != report['prediction_sha256'][score_path.name]):
        raise ValueError('incomplete or changed OOF baseline')
    entries = select_parts(manifest, count)
    users, directions, observed, latest = train(raw, prepared, entries)
    if len(users) == 0:
        raise ValueError('no train feeds with both positive and negative embedded candidates')
    val_parts = manifest['splits']['val']['parts']
    timing = pl.concat([pl.read_parquet(prepared / p['feed_path'], columns=['feed_id', 'timestamp']) for p in val_parts])
    if (latest >= timing['timestamp'].min() or timing['feed_id'].is_duplicated().any()
            or observed.intersection(timing['feed_id'])):
        raise ValueError('train and validation must contain separate chronological feeds')
    public, _ = pseudo_split(timing['feed_id'], timing['timestamp'])
    subset = timing.filter(pl.col('feed_id').is_in(public))
    early, late = pseudo_split(subset['feed_id'], subset['timestamp'], .5)
    scores = predict(raw, prepared, val_parts, public, users, directions, fit_ridge(users, directions))
    base = pl.read_parquet(score_path).filter(pl.col('feed_id').is_in(public)).rename({'target': 'base_target'})
    frame = scores.join(base, on=KEY, how='left', validate='1:1')
    if frame.height != base.height or frame.null_count().to_numpy().any() or not frame['target'].equals(frame['base_target']):
        raise ValueError('OOF keys or labels differ')
    return dict(model=model.name, prediction_sha256=report['prediction_sha256'][score_path.name],
                train_sources=[p['source'] for p in entries], train_feeds=len(observed), training_feeds_used=len(users),
                early_feeds=len(early), late_feeds=len(late), weights=WEIGHTS, metrics=evaluate(frame, early, late))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=Path('competition'))
    parser.add_argument('--prepared', type=Path, default=Path('data/prepared'))
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--train-parts', type=int, default=8)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = probe(args.raw, args.prepared, args.model, args.train_parts)
    write_json(args.out, result)
    print(json.dumps(result['metrics'], indent=2))
