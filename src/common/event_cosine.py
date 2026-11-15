"""Candidate/history cosine conditional on selected event types."""
import hashlib
import inspect
import json
import math
from pathlib import Path

import numpy as np
import polars as pl

from src.common.feature_cache import KEY, build_cache, selected_parts
from src.common.preparation import fingerprint, flat_values
from src.common.similarity import gather, padded_indices, prefetch_vectors, remap_indices, unit

EVENTS = (1, 2, 10)
WINDOW = 50
BATCH = 16


def feature_names():
    return [f'{prefix}_evt{event}_{suffix}_50' for event in EVENTS
            for prefix, suffix in (('hist', 'valid'), ('cos', 'mean'), ('cos', 'max'))]


def domains():
    return {name: (0, WINDOW) if name.startswith('hist_') else (-1, 1) for name in feature_names()}


def vector_identity(raw):
    result = {}
    for split in ('train', 'val', 'test'):
        directory = Path(raw) / split / 'item_embeddings'
        metadata = (directory / 'data.meta').read_bytes()
        meta = json.loads(metadata)
        data = fingerprint(directory / 'data.memmap')
        if (meta['dtype'] != 'int8' or meta['order'] != 'C' or meta['shape'][1] != 128
                or data['bytes'] != math.prod(meta['shape'])):
            raise ValueError('invalid original split vector file')
        result[split] = {'meta_sha256': hashlib.sha256(metadata).hexdigest(), 'data': data}
    shared = '\n'.join(inspect.getsource(function) for function in
                       (unit, gather, padded_indices, prefetch_vectors, remap_indices))
    return {'vectors': result, 'shared_math_sha256': hashlib.sha256(shared.encode()).hexdigest(),
            'events': list(EVENTS), 'window': WINDOW, 'batch_size': BATCH}


def required_rows(candidates, feeds):
    values = flat_values(feeds['history__item_id__embedding_idx'])
    lengths = feeds['history__item_id__embedding_idx'].list.len().to_numpy()
    offsets = np.r_[0, lengths.cumsum(dtype=np.int64)]
    recent, _ = padded_indices(values, offsets, 0, feeds.height, WINDOW)
    candidate = candidates['item_id__embedding_idx'].to_numpy()
    if (candidate < -1).any() or (recent < -1).any():
        raise ValueError('embedding index below -1')
    return np.unique(np.concatenate((candidate[candidate >= 0], recent[recent >= 0])))


def build_feature_part(candidates, feeds, vectors, vector_rows):
    ordered = candidates.sort('feed_id', 'candidate_index')
    sizes = ordered.group_by('feed_id', maintain_order=True).len()
    feeds = feeds.with_columns(pl.col('feed_id').cast(ordered.schema['feed_id']))
    joined = sizes.join(feeds, on='feed_id', how='left', validate='1:1')
    required = ['history__item_id__embedding_idx', 'history__event_type']
    if joined.select(required).null_count().to_numpy().any():
        raise ValueError('missing history for candidate feed')
    lengths = joined[required[0]].list.len().to_numpy()
    if not np.array_equal(lengths, joined[required[1]].list.len().to_numpy()):
        raise ValueError('history event/vector arrays are unaligned')
    candidate_lengths = joined['len'].to_numpy()
    candidate_offsets = np.r_[0, candidate_lengths.cumsum(dtype=np.int64)]
    history_offsets = np.r_[0, lengths.cumsum(dtype=np.int64)]
    candidate_values = remap_indices(ordered['item_id__embedding_idx'].to_numpy(), vector_rows)
    history_values = flat_values(joined[required[0]])
    event_values = flat_values(joined[required[1]])
    chunks = {name: [] for name in feature_names()}
    for start in range(0, joined.height, BATCH):
        stop = min(start + BATCH, joined.height)
        width = int(candidate_lengths[start:stop].max())
        candidate, present = padded_indices(candidate_values, candidate_offsets, start, stop, width)
        history, _ = padded_indices(history_values, history_offsets, start, stop, WINDOW)
        events, _ = padded_indices(event_values, history_offsets, start, stop, WINDOW)
        history = remap_indices(history, vector_rows)
        candidate_vectors = unit(gather(vectors, candidate))
        history_vectors = unit(gather(vectors, history))
        similarity = np.clip(np.matmul(candidate_vectors, history_vectors.transpose(0, 2, 1)), -1., 1.)
        for event in EVENTS:
            valid = (history >= 0) & (events == event)
            count = valid.sum(axis=1)
            event_result = {
                f'hist_evt{event}_valid_50': np.broadcast_to(count[:, None], candidate.shape),
                f'cos_evt{event}_mean_50': (similarity * valid[:, None, :]).sum(axis=2) / np.maximum(count[:, None], 1),
                f'cos_evt{event}_max_50': np.where(count[:, None] > 0,
                    np.where(valid[:, None, :], similarity, -np.inf).max(axis=2), 0.),
            }
            for name, field_values in event_result.items():
                chunks[name].append(field_values[present].astype(np.float32))
    return ordered.select(*KEY, 'item_id').with_columns([
        pl.Series(name, np.concatenate(values), dtype=pl.Float32) for name, values in chunks.items()])


def build_features(raw, prepared, output, *, train_parts=8, prefetch_parts=4, resume=False):
    raw, prepared, output = map(Path, (raw, prepared, output))
    if not 1 <= prefetch_parts <= 8:
        raise ValueError('prefetch_parts must be between 1 and 8')
    if raw.resolve() == output.resolve() or raw.resolve() in output.resolve().parents:
        raise ValueError('output must be separate from original data')
    source = json.loads((prepared / 'manifest.json').read_text())
    entries = {'train': selected_parts(source, train_parts),
               **{split: source['splits'][split]['parts'] for split in ('val', 'test')}}
    positions = {split: {entry['source']: index for index, entry in enumerate(parts)} for split, parts in entries.items()}
    loaded = {'split': None, 'frames': {}, 'vectors': None, 'rows': None}
    candidate_columns = [*KEY, 'item_id', 'item_id__embedding_idx']
    feed_columns = ['feed_id', 'history__item_id__embedding_idx', 'history__event_type']

    def build_part(split, entry, state):
        if loaded['split'] != split or entry['source'] not in loaded['frames']:
            first = positions[split][entry['source']]
            group = entries[split][first:first + prefetch_parts]
            frames = {}
            for part in group:
                candidates = pl.read_parquet(prepared / part['candidate_path'], columns=candidate_columns)
                feeds = pl.read_parquet(prepared / part['feed_path'], columns=feed_columns)
                frames[part['source']] = (candidates, feeds)
            rows = np.unique(np.concatenate([required_rows(candidates, feeds) for candidates, feeds in frames.values()]))
            loaded.update(split=split, frames=frames, rows=rows, vectors=prefetch_vectors(raw, split, rows))
        candidates, feeds = loaded['frames'].pop(entry['source'])
        result = build_feature_part(candidates, feeds, loaded['vectors'], loaded['rows'])
        if not loaded['frames']:
            loaded.update(vectors=None, rows=None)
        return result

    return build_cache(prepared, output, train_parts=train_parts, group='event_cosine', names=feature_names(),
                       domains=domains(), producer=__file__, extra=vector_identity(raw), build_part=build_part,
                       resume=resume, verify_extra=lambda: vector_identity(raw))
