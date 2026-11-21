"""Cosine against position-decayed history and its normalized vector profile."""
import json
from pathlib import Path

import numpy as np
import polars as pl

from src.common.event_cosine import BATCH, required_rows, vector_identity
from src.common.feature_cache import KEY, build_cache, selected_parts
from src.common.preparation import flat_values
from src.common.similarity import gather, padded_indices, prefetch_vectors, remap_indices, unit

WINDOWS = (10, 50)
HALF_LIFE = 10.


def feature_names():
    return [f'rc_cos_{kind}_{window}' for window in WINDOWS for kind in ('mean', 'profile')]


def domains():
    return {name: (-1, 1) for name in feature_names()}


def dependency_identity(raw):
    result = vector_identity(raw)
    result.update(half_life_positions=HALF_LIFE, windows=list(WINDOWS), weighting='original history positions; missing vectors retain their positions')
    return result


def build_feature_part(candidates, feeds, vectors, vector_rows):
    ordered = candidates.sort('feed_id', 'candidate_index')
    if not ordered.height:
        return ordered.select(*KEY, 'item_id').with_columns([pl.Series(name, [], dtype=pl.Float32) for name in feature_names()])
    sizes = ordered.group_by('feed_id', maintain_order=True).len()
    feeds = feeds.with_columns(pl.col('feed_id').cast(ordered.schema['feed_id']))
    joined = sizes.join(feeds, on='feed_id', how='left', validate='1:1')
    if joined['history__item_id__embedding_idx'].null_count():
        raise ValueError('missing query histories')
    history_lengths = joined['history__item_id__embedding_idx'].list.len().to_numpy()
    lengths = joined['len'].to_numpy()
    candidate_offsets = np.r_[0, lengths.cumsum(dtype=np.int64)]
    history_offsets = np.r_[0, history_lengths.cumsum(dtype=np.int64)]
    candidate_values = remap_indices(ordered['item_id__embedding_idx'].to_numpy(), vector_rows)
    history_values = flat_values(joined['history__item_id__embedding_idx'])
    decay = np.exp2(-np.arange(max(WINDOWS), dtype=np.float64) / HALF_LIFE)
    chunks = {name: [] for name in feature_names()}
    for start in range(0, joined.height, BATCH):
        stop = min(start + BATCH, joined.height)
        candidates, present = padded_indices(candidate_values, candidate_offsets, start, stop, int(lengths[start:stop].max()))
        history, _ = padded_indices(history_values, history_offsets, start, stop, max(WINDOWS))
        history = remap_indices(history, vector_rows)
        candidate_vectors, history_vectors = unit(gather(vectors, candidates)), unit(gather(vectors, history))
        similarity = np.clip(np.matmul(candidate_vectors, history_vectors.transpose(0, 2, 1)), -1., 1.)
        for window in WINDOWS:
            weights = decay[:window][None, :] * (history[:, :window] >= 0)
            mean = (similarity[:, :, :window] * weights[:, None, :]).sum(axis=2) / np.maximum(weights.sum(axis=1)[:, None], 1e-30)
            profile = unit(np.einsum('bh,bhd->bd', weights, history_vectors[:, :window]))
            cosine = np.clip(np.einsum('bmd,bd->bm', candidate_vectors, profile), -1., 1.)
            chunks[f'rc_cos_mean_{window}'].append(mean[present].astype(np.float32))
            chunks[f'rc_cos_profile_{window}'].append(cosine[present].astype(np.float32))
    return ordered.select(*KEY, 'item_id').with_columns([
        pl.Series(name, np.concatenate(values), dtype=pl.Float32) for name, values in chunks.items()])


def build_features(raw, prepared, output, *, train_parts=8, prefetch_parts=2, resume=False, train_start=None):
    raw, prepared, output = map(Path, (raw, prepared, output))
    if not 1 <= prefetch_parts <= 8:
        raise ValueError('prefetch parts must be between 1 and 8')
    source = json.loads((prepared / 'manifest.json').read_text())
    entries = {'train': selected_parts(source, train_parts, train_start),
               **{split: source['splits'][split]['parts'] for split in ('val', 'test')}}
    positions = {split: {p['source']: i for i, p in enumerate(parts)} for split, parts in entries.items()}
    loaded = {'split': None, 'frames': {}, 'rows': None, 'vectors': None}

    def build_part(split, entry, state):
        if loaded['split'] != split or entry['source'] not in loaded['frames']:
            first = positions[split][entry['source']]
            frames = {}
            for part in entries[split][first:first + prefetch_parts]:
                candidates = pl.read_parquet(prepared / part['candidate_path'], columns=[*KEY, 'item_id', 'item_id__embedding_idx'])
                feeds = pl.read_parquet(prepared / part['feed_path'], columns=['feed_id', 'history__item_id__embedding_idx'])
                frames[part['source']] = candidates, feeds
            rows = np.unique(np.concatenate([required_rows(c, h) for c, h in frames.values()]))
            loaded.update(split=split, frames=frames, rows=rows, vectors=prefetch_vectors(raw, split, rows))
        candidates, feeds = loaded['frames'].pop(entry['source'])
        result = build_feature_part(candidates, feeds, loaded['vectors'], loaded['rows'])
        if not loaded['frames']:
            loaded.update(rows=None, vectors=None)
        return result

    return build_cache(prepared, output, train_parts=train_parts, group='recency_cosine', names=feature_names(),
        domains=domains(), producer=__file__, extra=dependency_identity(raw), build_part=build_part,
        resume=resume, train_start=train_start, verify_extra=lambda: dependency_identity(raw))
