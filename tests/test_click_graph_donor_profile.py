import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl

from scripts.profile_click_graph_donors import profile


class DonorProfileTests(unittest.TestCase):
    def test_counts_only_donors_and_positive_history_ids(self):
        with TemporaryDirectory() as tmp:
            prepared = Path(tmp)
            splits = {s: {"parts": []} for s in ("train", "val", "test")}
            for name, split, feed, timestamp, candidates, targets in [
                ("a", "train", 1, 10, [42], [1]),
                ("b", "train", 2, 11, [42, 44], [0, 1]),
                ("v", "val", 3, 12, [42], None),
                ("t", "test", 4, 13, [47], None),
            ]:
                cp, fp = f"{name}-c.parquet", f"{name}-f.parquet"
                frame = pl.DataFrame({"feed_id": [feed] * len(candidates), "item_id": candidates})
                if targets is not None:
                    frame = frame.with_columns(pl.Series("target", targets))
                frame.write_parquet(prepared / cp)
                pl.DataFrame(
                    {"feed_id": [feed], "timestamp": [timestamp], "history__item_id": [[10, -1, 12]]}
                ).write_parquet(prepared / fp)
                splits[split]["parts"].append(
                    {
                        "source": name,
                        "feed_path": fp,
                        "candidate_path": cp,
                        "candidates": len(candidates),
                        "feeds": 1,
                    }
                )
            (prepared / "manifest.json").write_text(json.dumps({"complete": True, "splits": splits}))
            report = profile(prepared, 1, 0)
            self.assertEqual(report["excluded_shards"], ["a"])
            self.assertEqual(
                report["totals"],
                {
                    "feeds": 1,
                    "impressions": 2,
                    "clicks": 1,
                    "query_impressions": 1,
                    "query_clicks": 0,
                    "history_impression_pairs": 2,
                    "history_click_pairs": 0,
                },
            )
            val = prepared / "v-f.parquet"
            pl.read_parquet(val).with_columns(pl.lit(11).alias("timestamp")).write_parquet(val)
            with self.assertRaisesRegex(ValueError, "chronology"):
                profile(prepared, 1, 0)
