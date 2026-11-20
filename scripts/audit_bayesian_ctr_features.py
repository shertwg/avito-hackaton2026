"""Audit donor CTR with independent daily counts from original prepared rows."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time

import numpy as np
import polars as pl

from scripts.feature_audit import audit_cache, sample_candidates
from src.common.bayesian_ctr import WINDOWS, domains, feature_names
from src.common.feature_cache import KEY
from src.common.preparation import write_json
from src.common.resources import memory_snapshot


def audit(prepared, cache, output):
    started = time.monotonic()
    prepared, cache, output = map(Path, (prepared, cache, output))
    source = json.loads((prepared / 'manifest.json').read_text())
    manifest = json.loads((cache / 'manifest.json').read_text())
    excluded = set(manifest['excluded_train_sources'])
    items = set()
    for split, info in manifest['splits'].items():
        for entry in source['splits'][split]['parts']:
            if entry['source'] in info['parts']:
                candidates = pl.read_parquet(prepared / entry['candidate_path'], columns=[*KEY, 'item_id']).sort('feed_id', 'candidate_index')
                items.update(sample_candidates(candidates)['item_id'].to_list())
    item_counts, category_counts, global_counts = [defaultdict(lambda: [0, 0]) for _ in range(3)]
    donors = [p for p in source['splits']['train']['parts'] if p['source'] not in excluded]
    selected_items = list(items)
    for number, entry in enumerate(donors, 1):
        frame = pl.read_parquet(prepared / entry['candidate_path'], columns=['item_id', 'category_id', 'timestamp', 'target'])
        frame = frame.with_columns((pl.col('timestamp').cast(pl.Int64) // 86400).cast(pl.Int32).alias('day'))
        categories = frame.group_by('category_id', 'day').agg(pl.len().alias('n'), pl.col('target').cast(pl.Int64).sum().alias('k'))
        for category, day, n, k in categories.iter_rows():
            category_counts[category, day][0] += n
            category_counts[category, day][1] += k
            global_counts[day][0] += n
            global_counts[day][1] += k
        selected = frame.filter((pl.col('item_id') > 0) & pl.col('item_id').is_in(selected_items))
        selected = selected.group_by('item_id', 'category_id', 'day').agg(pl.len().alias('n'), pl.col('target').cast(pl.Int64).sum().alias('k'))
        for item, category, day, n, k in selected.iter_rows():
            item_counts[item, category, day][0] += n
            item_counts[item, category, day][1] += k
        if number % 128 == 0:
            print(f'bayesian CTR audit donors {number}/{len(donors)}', flush=True)
    smoothing = manifest['identity']['extra']['smoothing']
    last_day = max(global_counts, default=min(manifest['identity']['extra']['days']) - 1)
    statistics = json.loads((cache / '_stats.json').read_text())
    impressions = sum(v[0] for v in global_counts.values())
    clicks = sum(v[1] for v in global_counts.values())
    if impressions != statistics['donor_impressions'] or clicks != statistics['donor_clicks']:
        raise ValueError('independent donor totals disagree')
    computed = {}

    def reference(split, candidates, feeds, entry):
        rows = []
        for row in candidates.iter_rows(named=True):
            item, category, day = row['item_id'], row['category_id'], row['timestamp'] // 86400
            key = item, category, day
            if key not in computed:
                cutoff = min(day - 1, last_day)
                result = []
                for window in WINDOWS:
                    eligible = [d for d in global_counts if d <= cutoff and (window == 'all' or d >= cutoff - int(window[:-1]) + 1)]
                    n_global = sum(global_counts[d][0] for d in eligible)
                    k_global = sum(global_counts[d][1] for d in eligible)
                    n_cat = sum(category_counts.get((category, d), (0, 0))[0] for d in eligible)
                    k_cat = sum(category_counts.get((category, d), (0, 0))[1] for d in eligible)
                    n_item = sum(item_counts.get((item, category, d), (0, 0))[0] for d in eligible)
                    k_item = sum(item_counts.get((item, category, d), (0, 0))[1] for d in eligible)
                    prior = k_global / n_global if n_global else 0.
                    category_ctr = (k_cat + smoothing * prior) / (n_cat + smoothing)
                    item_ctr = (k_item + smoothing * category_ctr) / (n_item + smoothing)
                    result.extend((item_ctr, category_ctr, item_ctr - category_ctr, math.log1p(n_item)))
                eligible = [d for d in global_counts if d <= cutoff]
                result.extend((day - max(eligible) if eligible else 0., float(bool(eligible))))
                computed[key] = result
            rows.append(computed[key])
        return np.asarray(rows, dtype=np.float32).reshape(-1, len(feature_names()))

    report = audit_cache(prepared, cache, output, group='bayesian_ctr', names=feature_names(), domains=domains(),
        reference=reference, producer=__file__, candidate_columns=['timestamp', 'category_id'])
    report.update(donor_parts=len(donors), donor_impressions=impressions, donor_clicks=clicks,
                  independently_counted_items=len(items), reference_scope='all donor category/global rows and sampled item/day counts',
                  sample_order='feed_id,candidate_index', duration_seconds=round(time.monotonic() - started, 2), memory=memory_snapshot())
    write_json(output, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, default=Path('data/prepared'))
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.prepared, args.cache, args.out), indent=2))
