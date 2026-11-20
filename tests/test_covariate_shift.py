import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl

from scripts.audit_covariate_shift import BASE, auc_score, classify, sample_part


class ShiftAuditTests(unittest.TestCase):
    def test_sampling_ignores_labels_and_row_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "part.parquet"
            frame = pl.DataFrame(
                {
                    "feed_id": [1, 1, 2, 2],
                    "user_id": [3, 3, 4, 4],
                    "candidate_index": [0, 1, 0, 1],
                    "item_id": [7, 8, 9, 10],
                    "target": [1, 0, 1, 0],
                    **{f: [0, 1, 2, 3] for f in BASE},
                }
            )
            entry = {"candidate_path": path.name, "feeds": 2}
            frame.write_parquet(path)
            original = sample_part(path.parent, entry, "val", [])
            frame.reverse().with_columns((1 - pl.col("target")).alias("target")).write_parquet(path)
            self.assertTrue(sample_part(path.parent, entry, "val", []).equals(original))
            self.assertNotIn("target", original.columns)

    def test_auc_and_negative_control(self):
        self.assertAlmostEqual(auc_score(np.array([0, 1, 0, 1]), np.array([0.0, 1.0, 1.0, 1.0])), 0.75)
        users = np.repeat(np.arange(120), 2)
        a = pl.DataFrame({"feed_id": np.arange(240), "user_id": users, "signal": np.zeros(240)})
        b = a.with_columns((pl.col("feed_id") + 240).alias("feed_id"))
        self.assertEqual(classify(a, b, ["signal"], trees=3)["auc"], 0.5)
        b = b.with_columns(pl.lit(1.0).alias("signal"))
        self.assertGreater(classify(a, b, ["signal"], trees=3)["auc"], 0.99)
