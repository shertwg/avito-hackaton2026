"""Audit UTC fields and sampled ages against every donor candidate shard."""
import argparse
import datetime
import json
from pathlib import Path
import time

import numpy as np
import polars as pl

from scripts.feature_audit import audit_cache, sample_candidates
from src.common.feature_cache import KEY
from src.common.preparation import write_json
from src.common.time_features import domains, feature_names


def audit(prepared, cache, output):
    prepared, cache = Path(prepared), Path(cache)
    started = time.monotonic()
    source = json.loads((prepared / 'manifest.json').read_text())
    manifest = json.loads((cache / 'manifest.json').read_text())
    donors = set(manifest['identity']['extra']['donor_sources'])
    excluded = set(manifest['excluded_train_sources'])
    all_train = {entry['source'] for entry in source['splits']['train']['parts']}
    if donors & excluded or donors | excluded != all_train:
        raise ValueError('time donor partition differs')
    items = set()
    for split, info in manifest['splits'].items():
        for entry in source['splits'][split]['parts']:
            if entry['source'] in info['parts']:
                frame = pl.read_parquet(prepared / entry['candidate_path'], columns=[*KEY, 'item_id']).sort('feed_id', 'candidate_index')
                items.update(item for item in sample_candidates(frame)['item_id'].to_list() if item > 0)
    first_seen = {}
    for number, entry in enumerate(source['splits']['train']['parts'], 1):
        if entry['source'] not in donors:
            continue
        frame = pl.read_parquet(prepared / entry['candidate_path'], columns=['item_id', 'timestamp'])
        values = frame.filter(pl.col('item_id').is_in(list(items))).group_by('item_id').agg(pl.col('timestamp').min())
        for item, timestamp in values.iter_rows():
            first_seen[item] = min(first_seen.get(item, timestamp), timestamp)
        if number % 128 == 0:
            print(f'time audit donor source: {number}', flush=True)
    epoch = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)

    def reference(split, candidates, feeds, entry):
        values = []
        for row in candidates.iter_rows(named=True):
            timestamp = row['timestamp']
            clock = epoch + datetime.timedelta(seconds=timestamp)
            first = first_seen.get(row['item_id'])
            known = first is not None and first < timestamp and row['item_id'] > 0
            values.append((clock.hour, clock.weekday(), (timestamp - first) / 86400. if known else 0., float(known)))
        return np.asarray(values, dtype=np.float32).reshape(-1, len(feature_names()))

    report = audit_cache(prepared, cache, output, group='time', names=feature_names(), domains=domains(),
                         reference=reference, producer=__file__, candidate_columns=['timestamp'])
    report.update(duration_seconds=round(time.monotonic() - started, 2), donor_parts=len(donors),
                  reference_item_count=len(items), age_reference='minimum timestamp over all donor shards for sampled items')
    write_json(Path(output), report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, default=Path('data/prepared'))
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    audit(args.prepared, args.cache, args.out)
