import tempfile
import unittest
from pathlib import Path

import polars as pl

from scripts.count_zero_click_feeds import count_groups


class EmptyFeedTests(unittest.TestCase):
    def test_all_negative_and_all_positive_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ("train", "val"):
                directory = root / split / "impressions"
                directory.mkdir(parents=True)
                pl.DataFrame({"feed_id": [1, 2, 3], "target": [[0, 0], [1, 1], [1, 0, 0]]}).write_parquet(
                    directory / "part-00000.parquet"
                )
            v = count_groups(root)["train"]
            self.assertEqual((v["zero_click_feeds"], v["zero_click_candidates"]), (1, 2))
            self.assertEqual((v["all_click_feeds"], v["all_click_candidates"]), (1, 2))
            self.assertEqual((v["feeds"], v["candidates"], v["clicks"]), (3, 7, 3))
