"""Audit full index structure and sampled item2item formulas from all donors."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import polars as pl

from scripts.feature_audit import audit_cache, sample_candidates
from src.common.feature_cache import KEY
from src.common.item2item import ItemHistoryIndex, RECENT, domains, feature_names
from src.common.preparation import write_json
from src.common.resources import memory_snapshot


def validate_index(index):
    for start in range(0, len(index.items), 1_000_000):
        stop = min(len(index.items), start + 1_000_000)
        values = index.items[max(0, start - 1):stop]
        if (values <= 0).any() or (values[1:] <= values[:-1]).any():
            raise ValueError('index items not strictly increasing and positive')
        offsets = index.offsets[start:stop + 1]
        if (offsets[1:] <= offsets[:-1]).any():
            raise ValueError('empty or reversed posting range')
    for start in range(0, len(index.feeds), 1_000_000):
        first = max(0, start - 1)
        stop = min(len(index.feeds), start + 1_000_000)
        values = index.feeds[first:stop]
        bad = np.flatnonzero(values[1:] <= values[:-1]) + first + 1
        positions = np.searchsorted(index.offsets, bad)
        if len(bad) and not np.array_equal(index.offsets[positions], bad):
            raise ValueError('posting list duplicated or not strictly sorted')


def audit(prepared, cache, output):
    prepared, cache, output = map(Path, (prepared, cache, output))
    started = time.monotonic()
    source_bytes = (prepared / 'manifest.json').read_bytes()
    manifest_bytes = (cache / 'manifest.json').read_bytes()
    source, manifest = json.loads(source_bytes), json.loads(manifest_bytes)
    if not source.get('complete') or not manifest.get('complete'):
        raise ValueError('complete source and item2item cache required')
    donors = set(manifest['identity']['extra']['donor_sources'])
    excluded = set(manifest['excluded_train_sources'])
    all_train = {entry['source'] for entry in source['splits']['train']['parts']}
    if donors & excluded or donors | excluded != all_train:
        raise ValueError('item2item donor partition differs')
    with ItemHistoryIndex(cache) as index:
        validate_index(index)
        index_meta = index.metadata
        expected_identity = hashlib.sha256(json.dumps(manifest['identity'], sort_keys=True).encode()).hexdigest()
        if index_meta['identity_sha256'] != expected_identity:
            raise ValueError('index identity differs')
    items = set()
    for split, info in manifest['splits'].items():
        for entry in source['splits'][split]['parts']:
            if entry['source'] not in info['parts']:
                continue
            candidates = pl.read_parquet(prepared / entry['candidate_path'], columns=[*KEY, 'item_id']).sort('feed_id', 'candidate_index')
            sample = sample_candidates(candidates)
            items.update(item for item in sample['item_id'].to_list() if item > 0)
            feeds = pl.read_parquet(prepared / entry['feed_path'], columns=['feed_id', 'history__item_id'])
            feeds = feeds.filter(pl.col('feed_id').is_in(sample['feed_id'].unique()))
            for (history,) in feeds.select('history__item_id').iter_rows():
                items.update(item for item in history[:max(RECENT)] if item > 0)
    reference_postings = {item: set() for item in items}
    selected_items = list(items)
    donor_feed_count = 0
    for number, entry in enumerate(source['splits']['train']['parts'], 1):
        if entry['source'] not in donors:
            continue
        feeds = pl.read_parquet(prepared / entry['feed_path'], columns=['feed_id', 'history__item_id'])
        donor_feed_count += feeds.height
        pairs = feeds.explode('history__item_id').rename({'history__item_id': 'item_id'})
        pairs = pairs.filter(pl.col('item_id').is_in(selected_items)).unique(subset=['item_id', 'feed_id'])
        for item, ids in pairs.group_by('item_id').agg('feed_id').iter_rows():
            reference_postings[item].update(ids)
        if number % 128 == 0:
            print(f'item2item audit donor source: {number}', flush=True)
    if donor_feed_count != index_meta['donor_feeds']:
        raise ValueError('donor feed count differs')
    with ItemHistoryIndex(cache) as index:
        ordered_items = sorted(items)
        for start in range(0, len(ordered_items), 1000):
            selected = ordered_items[start:start + 1000]
            for item, posting in zip(selected, index.posting_lists(selected)):
                if set(posting.tolist()) != reference_postings[item]:
                    raise ValueError('independent item posting mismatch')

    def reference(split, candidates, feeds, entry):
        histories = dict(feeds.select('feed_id', 'history__item_id').iter_rows())
        result = []
        for row in candidates.iter_rows(named=True):
            left = reference_postings.get(row['item_id'], set())
            values = []
            for item in histories[row['feed_id']][:max(RECENT)]:
                right = reference_postings.get(item, set())
                denominator = math.sqrt(len(left) * len(right))
                values.append(len(left.intersection(right)) / denominator if denominator else 0.)
            output_values = []
            for window in RECENT:
                recent = values[:window]
                output_values.extend((sum(recent), max(recent, default=0.)))
            result.append(output_values)
        return np.asarray(result, dtype=np.float32).reshape(-1, len(feature_names()))

    report = audit_cache(prepared, cache, output, group='item2item', names=feature_names(), domains=domains(),
                         reference=reference, producer=__file__, feed_columns=['history__item_id'])
    if ((prepared / 'manifest.json').read_bytes() != source_bytes
            or (cache / 'manifest.json').read_bytes() != manifest_bytes):
        raise ValueError('source changed during independent index audit')
    report.update(duration_seconds=round(time.monotonic() - started, 2), donor_parts=len(donors),
                  donor_feeds=donor_feed_count, index_items=index_meta['items'], index_postings=index_meta['postings'],
                  independently_verified_items=len(items), index_structure='all items, offsets and posting lists',
                  reference_scope='sampled candidate/history item postings independently collected from all donor histories',
                  memory=memory_snapshot())
    write_json(output, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, default=Path('data/prepared'))
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.prepared, args.cache, args.out), indent=2))
