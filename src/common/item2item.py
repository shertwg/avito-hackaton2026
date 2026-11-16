"""Exact item co-history features computed from a disk-backed inverted index."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
from scipy import sparse

from src.common.feature_cache import KEY, build_cache, selected_parts
from src.common.preparation import fingerprint, write_json

RECENT = (10, 50)
ARRAYS = ('_items.npy', '_offsets.npy', '_feeds.npy')


def feature_names():
    return [f'i2i_{operation}_{window}' for window in RECENT for operation in ('sum', 'max')]


def domains():
    return {f'i2i_{operation}_{window}': (0, window if operation == 'sum' else 1)
            for window in RECENT for operation in ('sum', 'max')}


def checkpoint_table(connection, name, query):
    connection.execute(f'CREATE TABLE IF NOT EXISTS {name} AS {query}')
    connection.execute('CHECKPOINT')
    print(f'item2item: {name} ready', flush=True)


def publish_arrays(output):
    for name in ARRAYS:
        os.replace(output / (name + '.tmp'), output / name)


def export_index(connection, output, identity_sha256, donor_feeds, history_events):
    count, postings = connection.execute('SELECT count(*),coalesce(sum(support),0) FROM item_support').fetchone()
    count, postings = int(count), int(postings)
    items = np.lib.format.open_memmap(output / '_items.npy.tmp', mode='w+', dtype=np.int32, shape=(count,))
    offsets = np.lib.format.open_memmap(output / '_offsets.npy.tmp', mode='w+', dtype=np.int64, shape=(count + 1,))
    offsets[0] = 0
    position, total = 0, 0
    reader = connection.execute('SELECT item_id,support FROM item_support ORDER BY item_id').fetch_record_batch(1_000_000)
    for batch in reader:
        ids = batch.column(0).to_numpy()
        support = batch.column(1).to_numpy().astype(np.int64)
        size = len(ids)
        items[position:position + size] = ids
        offsets[position + 1:position + size + 1] = total + support.cumsum(dtype=np.int64)
        total += int(support.sum())
        position += size
    if position != count or total != postings:
        raise ValueError('item support export count differs')
    items.flush()
    offsets.flush()
    del items, offsets
    feeds = np.lib.format.open_memmap(output / '_feeds.npy.tmp', mode='w+', dtype=np.int32, shape=(postings,))
    position = 0
    reader = connection.execute('SELECT feed_id FROM postings ORDER BY item_id,feed_id').fetch_record_batch(1_000_000)
    for batch in reader:
        values = batch.column(0).to_numpy()
        feeds[position:position + len(values)] = values
        position += len(values)
    if position != postings:
        raise ValueError('posting export count differs')
    feeds.flush()
    del feeds
    publish_arrays(output)
    metadata = {'complete': True, 'identity_sha256': identity_sha256, 'items': count, 'postings': postings,
                'donor_feeds': int(donor_feeds), 'history_events': int(history_events),
                'query_items': connection.execute('SELECT count(*) FROM query_items').fetchone()[0],
                'files': {name: fingerprint(output / name) for name in ARRAYS}}
    write_json(output / '_index.json', metadata)
    print(f'item2item index: {count:,} items, {postings:,} postings', flush=True)


def ensure_index(prepared, output, donors, queries, identity_sha256, memory_mb):
    marker = output / '_index.json'
    if marker.exists():
        meta = json.loads(marker.read_text())
        try:
            files_match = (set(meta.get('files', {})) == set(ARRAYS)
                           and all(fingerprint(output / name) == stamp for name, stamp in meta['files'].items()))
        except OSError:
            files_match = False
        if meta.get('complete') and meta.get('identity_sha256') == identity_sha256 and files_match:
            return
    connection = duckdb.connect(str(output / '_index.duckdb'), config={
        'memory_limit': f'{memory_mb}MB', 'threads': '1', 'preserve_insertion_order': 'false',
        'temp_directory': str(output / '_spill')})
    try:
        connection.from_parquet([str(prepared / entry['feed_path']) for entry in donors]).create_view('donor_feeds', replace=True)
        connection.from_parquet([str(prepared / entry['candidate_path']) for entry in queries]).create_view('query_candidates', replace=True)
        connection.from_parquet([str(prepared / entry['feed_path']) for entry in queries]).create_view('query_feeds', replace=True)
        invalid = connection.execute('SELECT count(*) FROM donor_feeds WHERE feed_id IS NULL '
            'OR history__item_id IS NULL OR len(history__item_id)!=list_count(history__item_id)').fetchone()[0]
        if invalid:
            raise ValueError('null donor feed or history element')
        donor_count, distinct_count, history_events = connection.execute(
            'SELECT count(*),count(DISTINCT feed_id),coalesce(sum(len(history__item_id)),0) FROM donor_feeds').fetchone()
        if donor_count != distinct_count:
            raise ValueError('duplicate donor feed identifiers')
        checkpoint_table(connection, 'query_items',
            'SELECT item_id::INTEGER item_id FROM query_candidates WHERE item_id>0 UNION '
            'SELECT item_id::INTEGER FROM (SELECT unnest(history__item_id[1:50]) item_id FROM query_feeds) WHERE item_id>0')
        checkpoint_table(connection, 'postings',
            'SELECT DISTINCT item_id::INTEGER item_id,feed_id::INTEGER feed_id '
            'FROM (SELECT feed_id,unnest(history__item_id) item_id FROM donor_feeds) h '
            'SEMI JOIN query_items USING (item_id) WHERE item_id>0')
        checkpoint_table(connection, 'item_support',
            'SELECT item_id,count(*)::BIGINT support FROM postings GROUP BY item_id')
        export_index(connection, output, identity_sha256, donor_count, history_events)
    finally:
        connection.close()


class ItemHistoryIndex:
    def __init__(self, root):
        self.root = Path(root)
        self.metadata = json.loads((self.root / '_index.json').read_text())
        if not self.metadata.get('complete') or set(self.metadata['files']) != set(ARRAYS):
            raise ValueError('item index incomplete')
        for name, stamp in self.metadata['files'].items():
            if fingerprint(self.root / name) != stamp:
                raise ValueError('item index file changed')
        self.items = np.load(self.root / '_items.npy', mmap_mode='r', allow_pickle=False)
        self.offsets = np.load(self.root / '_offsets.npy', mmap_mode='r', allow_pickle=False)
        self.feeds = np.load(self.root / '_feeds.npy', mmap_mode='r', allow_pickle=False)
        if (self.items.dtype != np.int32 or self.offsets.dtype != np.int64 or self.feeds.dtype != np.int32
                or self.items.ndim != 1 or self.offsets.shape != (len(self.items) + 1,)
                or self.feeds.ndim != 1 or self.offsets[0] != 0 or self.offsets[-1] != len(self.feeds)
                or len(self.items) != self.metadata['items'] or len(self.feeds) != self.metadata['postings']):
            self.close()
            raise ValueError('invalid item index layout')

    def close(self):
        for name in ('items', 'offsets', 'feeds'):
            array = getattr(self, name, None)
            if array is not None and getattr(array, '_mmap', None) is not None:
                array._mmap.close()

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()

    def posting_lists(self, values):
        values = np.asarray(values, dtype=np.int32)
        positions = np.searchsorted(self.items, values)
        possible = (values > 0) & (positions < len(self.items))
        matched = np.zeros(len(values), dtype=bool)
        matched[possible] = self.items[positions[possible]] == values[possible]
        empty = np.empty(0, dtype=np.int32)
        return [self.feeds[self.offsets[position]:self.offsets[position + 1]] if found else empty
                for position, found in zip(positions, matched)]

    def similarities(self, candidates, history):
        values, inverse = np.unique(np.concatenate((candidates, history)), return_inverse=True)
        postings = self.posting_lists(values)
        lengths = np.array([len(posting) for posting in postings], dtype=np.int64)
        if not lengths.sum():
            return np.zeros((len(candidates), len(history)), dtype=np.float64)
        joined = np.concatenate(postings)
        donor_ids = np.unique(joined)
        local_columns = np.searchsorted(donor_ids, joined)
        matrix = sparse.csr_matrix((np.ones(len(joined), dtype=np.int32), local_columns,
                                    np.r_[0, lengths.cumsum(dtype=np.int64)]), shape=(len(values), len(donor_ids)))
        left, right = inverse[:len(candidates)], inverse[len(candidates):]
        counts = (matrix[left] @ matrix[right].T).toarray()
        denominator = np.sqrt(lengths[left, None].astype(np.float64) * lengths[right][None, :])
        return np.divide(counts, denominator, out=np.zeros_like(denominator), where=denominator > 0)


def build_feature_part(candidates, feeds, index, *, candidate_batch=64):
    if candidate_batch < 1:
        raise ValueError('candidate_batch must be positive')
    frame = candidates.sort('feed_id', 'candidate_index')
    if frame.null_count().to_numpy().any() or frame['item_id'].dtype != pl.Int32:
        raise ValueError('invalid candidate identifiers')
    if feeds['feed_id'].is_duplicated().any():
        raise ValueError('duplicate query histories')
    histories = dict(feeds.select('feed_id', 'history__item_id').iter_rows())
    ids = frame['feed_id'].to_numpy()
    item_ids = frame['item_id'].to_numpy()
    boundaries = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]
    output = np.zeros((frame.height, len(feature_names())), dtype=np.float32)
    for first, stop in zip(boundaries[:-1], boundaries[1:]):
        if first == stop:
            continue
        history = histories.get(int(ids[first]))
        if history is None or None in history:
            raise ValueError('missing query history or null item')
        history = np.asarray(history[:max(RECENT)], dtype=np.int64)
        if ((history < np.iinfo(np.int32).min).any() or (history > np.iinfo(np.int32).max).any()):
            raise ValueError('history item identifier exceeds Int32')
        history = history.astype(np.int32)
        if not len(history):
            continue
        for start in range(first, stop, candidate_batch):
            end = min(stop, start + candidate_batch)
            values = index.similarities(item_ids[start:end], history)
            for column, recent in enumerate(RECENT):
                output[start:end, column * 2] = values[:, :recent].sum(axis=1)
                output[start:end, column * 2 + 1] = values[:, :recent].max(axis=1)
    return frame.select(*KEY, 'item_id').with_columns([
        pl.Series(name, output[:, column], dtype=pl.Float32) for column, name in enumerate(feature_names())])


def build_features(prepared, output, *, train_parts=8, memory_mb=1536, candidate_batch=64, resume=False, first_train_shards=False):
    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    if memory_mb < 128 or candidate_batch < 1:
        raise ValueError('invalid item2item resource parameters')
    source = json.loads((prepared / 'manifest.json').read_text())
    selected = selected_parts(source, train_parts, first_train_shards=first_train_shards)
    excluded = {entry['source'] for entry in selected}
    donors = [entry for entry in source['splits']['train']['parts'] if entry['source'] not in excluded]
    if not donors:
        raise ValueError('item2item requires separate donor shards')
    queries = [*selected, *source['splits']['val']['parts'], *source['splits']['test']['parts']]

    @contextmanager
    def prepare():
        identity = json.loads((output / 'manifest.json').read_text())['identity']
        signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        ensure_index(prepared, output, donors, queries, signature, memory_mb)
        index = ItemHistoryIndex(output)
        try:
            yield index
        finally:
            index.close()

    def build_part(split, entry, index):
        candidates = pl.read_parquet(prepared / entry['candidate_path'], columns=[*KEY, 'item_id'])
        feeds = pl.read_parquet(prepared / entry['feed_path'], columns=['feed_id', 'history__item_id'])
        return build_feature_part(candidates, feeds, index, candidate_batch=candidate_batch)

    return build_cache(prepared, output, train_parts=train_parts, group='item2item', names=feature_names(),
                       domains=domains(), producer=__file__,
                       extra={'donor_sources': [entry['source'] for entry in donors], 'windows': list(RECENT),
                              'normalization': 'co-history feed presence cosine; self pairs included; nonpositive IDs unknown'},
                       build_part=build_part, prepare=prepare, resume=resume, excluded=excluded,
                       artifacts=('_index.json', *ARRAYS), first_train_shards=first_train_shards)
