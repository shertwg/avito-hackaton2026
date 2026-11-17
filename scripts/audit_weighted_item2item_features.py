"""Independently recompute weighted formulas on a previously audited item index."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from scripts.feature_audit import audit_cache
from src.common.item2item import ItemHistoryIndex
from src.common.preparation import write_json
from src.common.weighted_item2item import CHANNELS, HALF_LIFE, WINDOWS, dependency_identity, domains, feature_names


def audit(prepared, cache, parent, parent_audit, output):
    prepared, cache, parent, output = map(Path, (prepared, cache, parent, output))
    evidence = json.loads(Path(parent_audit).read_text())
    if (not evidence.get('complete') or evidence.get('group') != 'item2item'
            or evidence['manifest_sha256'] != hashlib.sha256((parent / 'manifest.json').read_bytes()).hexdigest()):
        raise ValueError('matching independent parent item2item audit required')
    source = json.loads((cache / 'manifest.json').read_text())
    train_parts = len(source['excluded_train_sources'])
    train_start = source['identity'].get('train_start')
    with ItemHistoryIndex(parent) as index:
        postings = {}

        def posting(item):
            if item not in postings:
                postings[item] = set(index.posting_lists([item])[0].tolist())
            return postings[item]

        def reference(split, candidates, feeds, entry):
            histories = {feed: (items, events) for feed, items, events in feeds.select(
                'feed_id', 'history__item_id', 'history__event_type').iter_rows()}
            rows = []
            for row in candidates.iter_rows(named=True):
                history, events = histories[row['feed_id']]
                left = posting(row['item_id'])
                result = []
                for window in WINDOWS:
                    for channel in CHANNELS:
                        weighted, total = [], 0.
                        for position, (item, event) in enumerate(zip(history[:window], events[:window])):
                            weight = 2. ** (-position / HALF_LIFE) if channel == 'all' or event == int(channel[3:]) else 0.
                            right = posting(item)
                            denominator = math.sqrt(len(left) * len(right))
                            score = len(left & right) / denominator if denominator else 0.
                            weighted.append(weight * score)
                            total += weight
                        result.extend((sum(weighted) / total if total else 0., max(weighted, default=0.)))
                rows.append(result)
            return np.asarray(rows, dtype=np.float32).reshape(-1, len(feature_names()))

        report = audit_cache(prepared, cache, output, group='weighted_item2item', names=feature_names(),
            domains=domains(), producer=__file__, reference=reference,
            feed_columns=['history__item_id', 'history__event_type'],
            verify_extra=lambda: dependency_identity(prepared, parent, train_parts, train_start))
    report.update(parent_audit_sha256=hashlib.sha256(Path(parent_audit).read_bytes()).hexdigest(),
        parent_manifest_sha256=evidence['manifest_sha256'],
        reference_policy='scalar event/position weighting and posting-set intersections on the independently audited parent index')
    write_json(output, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, default=Path('data/prepared'))
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--parent-audit', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.prepared, args.cache, args.parent, args.parent_audit, args.out), indent=2))
