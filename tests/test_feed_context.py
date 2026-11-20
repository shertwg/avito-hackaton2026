from pathlib import Path
import tempfile
import unittest

import numpy as np
import polars as pl

from scripts.audit_feed_context_features import audit
from src.common.feed_context import INPUTS, add_fields, build_features, feature_names
from src.common.item2item import build_features as build_item2item
from tests.new_feature_fixture import make_prepared, rank_sources


class FeedContextTests(unittest.TestCase):
    def test_ties_singletons_interactions_and_extreme_floats(self):
        frame = pl.DataFrame({'split': ['val'] * 4, 'feed_id': [1, 1, 1, 2], 'candidate_index': [0, 1, 2, 0],
            'item_id': [7, 7, 8, 9], **{name: [1., 1., 3., 2.] for name in INPUTS}},
            schema_overrides={name: pl.Float32 for name in INPUTS})
        frame = frame.with_columns(pl.lit(.2).cast(pl.Float32).alias('is_item_ctr'),
            pl.lit(.5).cast(pl.Float32).alias('cos_user'), pl.lit(-.5).cast(pl.Float32).alias('cos_hist_max_50'))
        result = add_fields(frame)
        self.assertEqual(len(feature_names()), 34)
        np.testing.assert_allclose(result['fc_i2i_sum_10_rank'], [.5, .5, 1., 1.])
        self.assertEqual(result['fc_cos_user_z'][3], 0.)
        self.assertAlmostEqual(result['fc_cos_user_x_item_ctr'][0], .1, places=6)
        self.assertAlmostEqual(result['fc_cos_hist_max_50_x_item_ctr'][0], -.1, places=6)
        self.assertEqual(result['fc_f1_gap_max'][0], -2.)
        extreme = frame.with_columns(pl.Series('f1', [np.finfo(np.float32).max, -np.finfo(np.float32).max, 0., 1.], dtype=pl.Float32))
        self.assertTrue(np.isfinite(add_fields(extreme).select(feature_names()).to_numpy()).all())
        with self.assertRaisesRegex(ValueError, 'finite'):
            add_fields(frame.with_columns(pl.lit(float('nan')).alias('f1')))

    def test_materialization_parent_cohorts_audit_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, prepared, source = make_prepared(root)
            cosine, itemstats = rank_sources(prepared, source, root)
            parent = root / 'item2item'
            build_item2item(prepared, parent, train_parts=2, memory_mb=256)
            output = root / 'context'
            first = build_features(prepared, cosine, itemstats, parent, output, train_parts=2)
            self.assertTrue(first['complete'])
            self.assertTrue(audit(prepared, cosine, itemstats, parent, output, root / 'audit.json')['complete'])
            part = output / 'train' / source['splits']['train']['parts'][0]['source']
            stamp = part.stat().st_mtime_ns
            build_features(prepared, cosine, itemstats, parent, output, train_parts=2, resume=True)
            self.assertEqual(part.stat().st_mtime_ns, stamp)
            with self.assertRaisesRegex(ValueError, 'cohort differs'):
                build_features(prepared, cosine, itemstats, parent, root / 'wrong', train_parts=1)


if __name__ == '__main__':
    unittest.main()
