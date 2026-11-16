"""Independently audit feed context with scalar ranks and NumPy statistics."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.feature_audit import audit_cache
from src.common.feed_context import BASE_FIELDS, SIGNALS, dependency_identity, domains, feature_names
from src.common.feature_cache import KEY
from src.common.feature_ranks import source_frame


def audit(prepared, cosine, itemstats, item2item, cache, output):
    manifest = json.loads((Path(cache) / 'manifest.json').read_text())
    count = len(manifest['excluded_train_sources'])

    def reference(split, candidates, feeds, entry):
        frame = candidates
        for root, fields in ((cosine, ['cos_user', 'cos_hist_max_50']), (itemstats, ['is_item_ctr']), (item2item, SIGNALS[2:])):
            values = source_frame(prepared, root, split, entry, fields)
            frame = frame.join(values, on=[*KEY, 'item_id'], how='left', validate='1:1')
        rows = []
        for group in frame.sort('feed_id', 'candidate_index').partition_by('feed_id', maintain_order=True):
            statistics = {name: group[name].to_numpy().astype(np.float64) for name in [*BASE_FIELDS, *SIGNALS]}
            for row in group.iter_rows(named=True):
                result = {}
                for name, values in statistics.items():
                    value = row[name]
                    result[f'fc_{name}_gap_max'] = value - max(values)
                    result[f'fc_{name}_gap_mean'] = value - float(values.mean())
                    if name in SIGNALS:
                        std = float(values.std(ddof=1)) if len(values) > 1 else 0.
                        result[f'fc_{name}_z'] = (value - float(values.mean())) / std if std else 0.
                        if name.startswith('i2i_'):
                            result[f'fc_{name}_rank'] = ((values < value).sum() + ((values == value).sum() + 1) / 2.) / len(values)
                result['fc_cos_user_x_item_ctr'] = row['cos_user'] * row['is_item_ctr']
                result['fc_cos_hist_max_50_x_item_ctr'] = row['cos_hist_max_50'] * row['is_item_ctr']
                bound = np.finfo(np.float32).max
                rows.append([np.clip(result[name], -bound, bound) for name in feature_names()])
        return np.asarray(rows, dtype=np.float32).reshape(-1, len(feature_names()))

    return audit_cache(prepared, cache, output, group='feed_context', names=feature_names(), domains=domains(),
        reference=reference, producer=__file__, candidate_columns=BASE_FIELDS,
        verify_extra=lambda: dependency_identity(prepared, cosine, itemstats, item2item, count))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, default=Path('data/prepared'))
    parser.add_argument('--cosine', type=Path, default=Path('data/features/cosine'))
    parser.add_argument('--itemstats', type=Path, required=True)
    parser.add_argument('--item2item', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.prepared, args.cosine, args.itemstats, args.item2item, args.cache, args.out), indent=2))
