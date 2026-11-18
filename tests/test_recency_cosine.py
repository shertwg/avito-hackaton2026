from pathlib import Path
import tempfile
import unittest

import numpy as np
import polars as pl

from scripts.audit_recency_cosine_features import audit
from src.common.recency_cosine import build_feature_part, build_features, feature_names
from tests.new_feature_fixture import make_prepared


class RecencyCosineTests(unittest.TestCase):
    def test_decay_original_positions_missing_and_duplicates(self):
        vectors = np.zeros((3, 128), dtype=np.int8)
        vectors[0, 0], vectors[1, 0], vectors[2, 1] = 1, -1, 1
        candidates = pl.DataFrame({'split': ['val'] * 5, 'feed_id': [1, 1, 1, 2, 3],
            'candidate_index': [0, 1, 2, 0, 0], 'item_id': [7, 7, 8, 9, 10],
            'item_id__embedding_idx': [0, 0, -1, 0, 0]})
        feeds = pl.DataFrame({'feed_id': [1, 2, 3],
            'history__item_id__embedding_idx': [[0, -1, 1], [], [-1] * 50 + [0]]})
        result = build_feature_part(candidates, feeds, vectors, np.arange(3))
        weight = 2. ** (-.2)
        self.assertAlmostEqual(result['rc_cos_mean_10'][0], (1 - weight) / (1 + weight), places=6)
        self.assertAlmostEqual(result['rc_cos_profile_50'][0], 1., places=6)
        self.assertEqual(result['rc_cos_mean_10'][0], result['rc_cos_mean_10'][1])
        self.assertTrue((result.tail(3).select(feature_names()).to_numpy() == 0).all())
        self.assertEqual(build_feature_part(candidates.head(0), feeds, vectors, np.arange(3)).height, 0)

    def test_materialization_audit_and_cohort_union_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, prepared, source = make_prepared(root)
            output = root / 'recency'
            first = build_features(raw, prepared, output, train_parts=3, prefetch_parts=1)
            path = output / 'train' / source['splits']['train']['parts'][0]['source']
            stamp = path.stat().st_mtime_ns
            second = build_features(raw, prepared, output, train_parts=2, prefetch_parts=2, resume=True)
            self.assertEqual(stamp, path.stat().st_mtime_ns)
            self.assertEqual(first['splits']['train']['covered_parts'], second['splits']['train']['covered_parts'])
            self.assertTrue(audit(raw, prepared, output, root / 'audit.json', train_parts=2)['complete'])


if __name__ == '__main__':
    unittest.main()
