"""Audit ranks using independent sorted-value tie boundaries."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.feature_audit import audit_cache
from src.common.feature_ranks import INPUTS, dependency_identity, domains, feature_names, source_frame


def audit(prepared, cosine, itemstats, cache, output):
    manifest = json.loads((Path(cache) / 'manifest.json').read_text())
    train_parts = len(manifest['excluded_train_sources'])
    def reference(split, candidates, feeds, entry):
        sources = []
        for root, columns in ((cosine, INPUTS[:3]), (itemstats, INPUTS[3:])):
            frame = source_frame(prepared, root, split, entry, columns)
            sources.append(frame.filter(frame['feed_id'].is_in(candidates['feed_id'].unique())).sort('feed_id', 'candidate_index'))
        keys = ['split', 'feed_id', 'candidate_index', 'item_id']
        for frame in sources:
            if not frame.select(keys).equals(candidates.select(keys)):
                raise ValueError('rank reference keys differ')
        expected = np.empty((candidates.height, len(INPUTS)), dtype=np.float32)
        ids = candidates['feed_id'].to_numpy()
        for feed in np.unique(ids):
            positions = np.flatnonzero(ids == feed)
            for column, name in enumerate(INPUTS):
                values = sources[0 if column < 3 else 1][name].to_numpy()[positions]
                ordered = np.sort(values)
                left = np.searchsorted(ordered, values, side='left')
                right = np.searchsorted(ordered, values, side='right')
                expected[positions, column] = (left + right + 1) / (2 * len(values))
        return expected
    return audit_cache(prepared, cache, output, group='feature_ranks', names=feature_names(), domains=domains(),
                       reference=reference, producer=__file__,
                       verify_extra=lambda: dependency_identity(prepared, cosine, itemstats, train_parts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, default=Path('data/prepared'))
    parser.add_argument('--cosine', type=Path, default=Path('data/features/cosine'))
    parser.add_argument('--itemstats', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    audit(args.prepared, args.cosine, args.itemstats, args.cache, args.out)
