import json
import tempfile
import unittest
from pathlib import Path

import polars as pl

from scripts.profile_prior_exposure import coverage


class ExposureCoverageTests(unittest.TestCase):
    def test_simultaneous_feeds_excluded_and_earlier_splits_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m = {"complete": True, "splits": {}}
            for split, rows in {
                "train": [(1, 1, 7, 21, 3), (8, 1, 0, 30, 7), (9, 1, 15, 0, 0)],
                "val": [(2, 2, 7, 22, 3), (3, 2, 7, 22, 3), (10, 2, 0, 30, 7), (11, 2, 15, 0, 0)],
                "test": [(4, 3, 7, 22, 3), (4, 3, 7, 21, 3), (4, 3, 7, 24, 4)],
            }.items():
                df = pl.DataFrame(
                    rows, schema=["feed_id", "timestamp", "user_id", "item_id", "category_id"], orient="row"
                )
                path = root / f"{split}.parquet"
                df.with_columns(pl.lit(split).alias("split")).write_parquet(path)
                m["splits"][split] = {
                    "parts": [
                        {
                            "candidate_path": path.name,
                            "feeds": df["feed_id"].n_unique(),
                            "candidates": len(rows),
                        }
                    ]
                }
            (root / "manifest.json").write_text(json.dumps(m))
            report = coverage(root, root)["coverage"]
            self.assertEqual(report["item_id"]["val"]["rows_with_previous"], 0)
            self.assertEqual(report["item_id"]["val"]["rows_with_missing_user_or_key"], 2)
            self.assertEqual(report["category_id"]["val"]["rows_with_previous"], 2)
            self.assertEqual(report["item_id"]["test"]["rows_with_previous"], 2)
            self.assertEqual(report["item_id"]["test"]["feeds_with_previous"], 1)
