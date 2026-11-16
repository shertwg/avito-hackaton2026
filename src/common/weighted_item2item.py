"""Recency- and event-weighted query histories on the audited co-history index."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

from src.common.feature_cache import KEY, build_cache, selected_parts
from src.common.item2item import ItemHistoryIndex
from src.common.preparation import fingerprint

WINDOWS = (10, 50)
CHANNELS = ('all', 'evt2', 'evt10')
HALF_LIFE = 10.


def feature_names():
    return [f'wi2i_{channel}_{operation}_{window}' for window in WINDOWS
            for channel in CHANNELS for operation in ('mean', 'max')]


def domains():
    return {name: (0, 1) for name in feature_names()}


def dependency_identity(prepared, parent, train_parts):
    source = json.loads((Path(prepared) / 'manifest.json').read_text())
    manifest = json.loads((Path(parent) / 'manifest.json').read_text())
    selected = {p['source'] for p in selected_parts(source, train_parts)}
    if not manifest.get('complete') or manifest.get('feature_group') != 'item2item':
        raise ValueError('complete item2item parent required')
    if selected != set(manifest['excluded_train_sources']):
        raise ValueError('weighted item2item cohort differs from parent')
    for name, stamp in manifest['artifacts'].items():
        if fingerprint(Path(parent) / name) != stamp:
            raise ValueError('parent item index changed')
    return {'parent_identity': manifest['identity'], 'parent_artifacts': manifest['artifacts'],
            'index_reader_sha256': hashlib.sha256(Path(__file__).with_name('item2item.py').read_bytes()).hexdigest(),
            'half_life_positions': HALF_LIFE, 'windows': list(WINDOWS), 'channels': list(CHANNELS),
            'weight_policy': 'event indicators on query history; position decay; normalize by full channel weight'}


def build_feature_part(candidates, feeds, index, *, candidate_batch=64):
    if candidate_batch < 1:
        raise ValueError('candidate batch must be positive')
    frame = candidates.sort('feed_id', 'candidate_index')
    if (frame.null_count().to_numpy().any() or frame['item_id'].dtype != pl.Int32
            or feeds['feed_id'].is_duplicated().any()):
        raise ValueError('invalid candidate IDs or duplicate histories')
    histories = {feed: (items, events) for feed, items, events in feeds.select(
        'feed_id', 'history__item_id', 'history__event_type').iter_rows()}
    feed_ids = frame['feed_id'].to_numpy()
    items = frame['item_id'].to_numpy()
    boundaries = np.r_[0, np.flatnonzero(feed_ids[1:] != feed_ids[:-1]) + 1, len(feed_ids)]
    values = np.zeros((frame.height, len(feature_names())), dtype=np.float32)
    for first, stop in zip(boundaries[:-1], boundaries[1:]):
        if first == stop:
            continue
        history, events = histories.get(int(feed_ids[first]), (None, None))
        if history is None or events is None or len(history) != len(events) or None in history or None in events:
            raise ValueError('unaligned or null history arrays')
        history = np.asarray(history[:max(WINDOWS)], dtype=np.int64)
        if ((history < np.iinfo(np.int32).min).any() or (history > np.iinfo(np.int32).max).any()):
            raise ValueError('history item exceeds Int32')
        events = np.asarray(events[:max(WINDOWS)])
        decay = np.exp2(-np.arange(len(history), dtype=np.float64) / HALF_LIFE)
        if not len(history):
            continue
        for start in range(first, stop, candidate_batch):
            end = min(stop, start + candidate_batch)
            similarity = index.similarities(items[start:end], history.astype(np.int32))
            column = 0
            for window in WINDOWS:
                for channel in CHANNELS:
                    mask = np.ones(len(history), dtype=bool) if channel == 'all' else events == int(channel[3:])
                    weights = (decay * mask)[:window]
                    weighted = similarity[:, :window] * weights[None, :]
                    values[start:end, column] = weighted.sum(axis=1) / max(weights.sum(), 1e-30)
                    values[start:end, column + 1] = weighted.max(axis=1)
                    column += 2
    return frame.select(*KEY, 'item_id').with_columns([
        pl.Series(name, values[:, i], dtype=pl.Float32) for i, name in enumerate(feature_names())])


def build_features(prepared, parent, output, *, train_parts=8, resume=False):
    prepared, parent, output = map(Path, (prepared, parent, output))
    extra = dependency_identity(prepared, parent, train_parts)
    selected = set(json.loads((parent / 'manifest.json').read_text())['excluded_train_sources'])

    @contextmanager
    def prepare():
        with ItemHistoryIndex(parent) as index:
            yield index

    def build_part(split, entry, index):
        candidates = pl.read_parquet(prepared / entry['candidate_path'], columns=[*KEY, 'item_id'])
        feeds = pl.read_parquet(prepared / entry['feed_path'], columns=['feed_id', 'history__item_id', 'history__event_type'])
        return build_feature_part(candidates, feeds, index)

    return build_cache(prepared, output, train_parts=train_parts, group='weighted_item2item',
        names=feature_names(), domains=domains(), producer=__file__, extra=extra, build_part=build_part,
        excluded=selected, resume=resume, prepare=prepare,
        verify_extra=lambda: dependency_identity(prepared, parent, train_parts))
