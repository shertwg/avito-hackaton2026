"""Audit event-conditioned cosine with independent Float64 vector calculations."""
import argparse
from pathlib import Path

import numpy as np

from scripts.feature_audit import audit_cache
from src.common.data import load_embeddings
from src.common.event_cosine import EVENTS, WINDOW, domains, feature_names, vector_identity


def audit(raw, prepared, cache, output, *, train_parts=None):
    def reference(split, candidates, feeds, entry):
        histories = {row['feed_id']: row for row in feeds.iter_rows(named=True)}
        vectors = load_embeddings(raw, split)
        expected = []
        try:
            for row in candidates.iter_rows(named=True):
                index = row['item_id__embedding_idx']
                if index < -1 or index >= len(vectors):
                    raise ValueError('invalid original candidate vector index')
                candidate = vectors[index].astype(np.float64) if index >= 0 else np.zeros(128)
                candidate /= max(np.linalg.norm(candidate), 1e-12)
                history = histories[row['feed_id']]
                indices, events = history['history__item_id__embedding_idx'], history['history__event_type']
                if len(indices) != len(events):
                    raise ValueError('unaligned original history')
                values = []
                for event in EVENTS:
                    similarities = []
                    for vector_index, event_type in zip(indices[:WINDOW], events[:WINDOW]):
                        if vector_index < -1 or vector_index >= len(vectors):
                            raise ValueError('invalid original history vector index')
                        if vector_index >= 0 and event_type == event:
                            vector = vectors[vector_index].astype(np.float64)
                            vector /= max(np.linalg.norm(vector), 1e-12)
                            similarities.append(float(np.clip(np.dot(candidate, vector), -1., 1.)))
                    values.extend((len(similarities), np.mean(similarities) if similarities else 0.,
                                   max(similarities, default=0.)))
                expected.append(values)
        finally:
            vectors._mmap.close()
        return np.asarray(expected, dtype=np.float32).reshape(-1, len(feature_names()))
    return audit_cache(prepared, cache, output, group='event_cosine', names=feature_names(), domains=domains(),
                       reference=reference, producer=__file__, candidate_columns=['item_id__embedding_idx'],
                       feed_columns=['history__item_id__embedding_idx', 'history__event_type'],
                       verify_extra=lambda: vector_identity(raw), train_parts=train_parts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=Path('competition'))
    parser.add_argument('--prepared', type=Path, default=Path('data/prepared'))
    parser.add_argument('--cache', type=Path, default=Path('data/features/event-cosine'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--train-parts', type=int)
    args = parser.parse_args()
    audit(args.raw, args.prepared, args.cache, args.out, train_parts=args.train_parts)
