"""Audit recency cosine independently in Float64 on original split vectors."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.feature_audit import audit_cache
from src.common.data import load_embeddings
from src.common.recency_cosine import HALF_LIFE, WINDOWS, dependency_identity, domains, feature_names


def audit(raw, prepared, cache, output, *, train_parts=None):
    def reference(split, candidates, feeds, entry):
        histories = dict(feeds.select('feed_id', 'history__item_id__embedding_idx').iter_rows())
        vectors = load_embeddings(raw, split)
        try:
            rows = []
            for row in candidates.iter_rows(named=True):
                index = row['item_id__embedding_idx']
                candidate = vectors[index].astype(np.float64) if index >= 0 else np.zeros(128)
                candidate /= max(np.linalg.norm(candidate), 1e-12)
                result = []
                for window in WINDOWS:
                    total, weighted, profile = 0., 0., np.zeros(128, dtype=np.float64)
                    for position, index in enumerate(histories[row['feed_id']][:window]):
                        if index < 0:
                            continue
                        vector = vectors[index].astype(np.float64)
                        vector /= max(np.linalg.norm(vector), 1e-12)
                        weight = 2. ** (-position / HALF_LIFE)
                        weighted += weight * np.clip(np.dot(candidate, vector), -1., 1.)
                        profile += weight * vector
                        total += weight
                    profile /= max(np.linalg.norm(profile), 1e-12)
                    result.extend((weighted / total if total else 0., np.clip(np.dot(candidate, profile), -1., 1.)))
                rows.append(result)
            return np.asarray(rows, dtype=np.float32).reshape(-1, len(feature_names()))
        finally:
            vectors._mmap.close()

    return audit_cache(prepared, cache, output, group='recency_cosine', names=feature_names(), domains=domains(),
        reference=reference, producer=__file__, candidate_columns=['item_id__embedding_idx'],
        feed_columns=['history__item_id__embedding_idx'], train_parts=train_parts,
        verify_extra=lambda: dependency_identity(raw))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=Path('competition'))
    parser.add_argument('--prepared', type=Path, default=Path('data/prepared'))
    parser.add_argument('--cache', type=Path, default=Path('data/features/recency-cosine'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--train-parts', type=int)
    args = parser.parse_args()
    print(json.dumps(audit(args.raw, args.prepared, args.cache, args.out, train_parts=args.train_parts), indent=2))
