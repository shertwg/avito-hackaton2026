import json
import tempfile
import unittest
from math import log1p
from pathlib import Path

import polars as pl

from scripts.audit_user_session_features import audit
from src.common.user_session import NAMES, build_features


def fixture(root):
    splits = {split: {"parts": []} for split in ("train", "val", "test")}
    for split, shard, feeds in [
        ("train", "donor", [(1, 0, 7, [(21, 3), (21, 3), (22, 4)]), (2, 0, 0, [(21, 3)])]),
        ("train", "chosen", [(3, 1800, 7, [(21, 4), (23, 3)]), (4, 1800, 7, [(24, 3)])]),
        (
            "val",
            "validation",
            [(5, 3600, 7, [(21, 3), (0, 0)]), (6, 5401, 7, [(21, 3)]), (7, 3600, 0, [(21, 3)])],
        ),
        ("test", "evaluation", [(8, 86400, 7, [(21, 4)]), (9, 86401, 7, [(21, 4)])]),
    ]:
        rows = []
        for feed, timestamp, user, candidates in feeds:
            for j, (item, category) in enumerate(candidates):
                row = {
                    "split": split,
                    "feed_id": feed,
                    "timestamp": timestamp,
                    "user_id": user,
                    "item_id": item,
                    "category_id": category,
                    "candidate_index": j,
                }
                if split != "test":
                    row["target"] = int(j == 0)
                if split == "train":
                    row["pos"] = j * 2
                rows.append(row)
        frame = pl.DataFrame(rows).with_columns(
            pl.col("feed_id", "item_id").cast(pl.Int32), pl.col("candidate_index").cast(pl.UInt32)
        )
        path = shard + ".parquet"
        frame.write_parquet(root / path)
        splits[split]["parts"].append(
            {
                "source": path,
                "candidate_path": path,
                "feed_path": path,
                "feeds": len(feeds),
                "candidates": len(rows),
            }
        )
    (root / "manifest.json").write_text(json.dumps({"complete": True, "splits": splits}))


class SessionTests(unittest.TestCase):
    def test_strict_history_and_donor_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = root / "prepared"
            prepared.mkdir()
            fixture(prepared)
            out = root / "first"
            m = build_features(prepared, out, train_parts=1, train_start=1)
            self.assertEqual(m["excluded_train_sources"], ["chosen.parquet"])
            tr = pl.read_parquet(out / "train/chosen.parquet")
            va = pl.read_parquet(out / "val/validation.parquet")
            te = pl.read_parquet(out / "test/evaluation.parquet")
            self.assertEqual(tr.select(NAMES[:5]).row(0), tr.select(NAMES[:5]).row(2))
            self.assertEqual(tr["us_item_prior_impr"][0], 1)
            self.assertAlmostEqual(tr["us_time_since_prev_feed_log"][0], log1p(1800), places=6)
            self.assertEqual(va["us_session_idx"].to_list(), [3, 3, 0, 0])
            self.assertEqual(va["us_feeds_last_1h"].to_list(), [3, 3, 1, 0])
            self.assertEqual(te["us_feeds_last_24h"].to_list(), [5, 5])
            self.assertEqual(va["us_item_prior_impr"].to_list(), [2, 0, 3, 0])
            self.assertAlmostEqual(va["us_cat_prior_impr_share"][0], 4 / 6, places=6)
            self.assertAlmostEqual(va["us_cat_recent3_impr_share"][2], 3 / 5, places=6)
            self.assertAlmostEqual(va["us_donor_user_ctr_smooth"][0], (1 + 5 * 0.5) / 8, places=6)
            self.assertEqual(va["us_donor_user_impr_log"][-1], 0)
            self.assertEqual(va["us_time_since_prev_feed_log"][-1], -1)
            self.assertEqual(va["us_donor_user_coe"][-1], 1)
            self.assertEqual(
                build_features(prepared, out, train_parts=1, train_start=1, resume=True)["features"], NAMES
            )
            evidence = audit(prepared, out, root / "audit.json")
            self.assertGreater(evidence["minimum_gap_seconds_in_checked_histories"], 0)
            self.assertGreater(evidence["sampled_candidates"], 0)
            for name in ("chosen", "validation"):
                path = prepared / f"{name}.parquet"
                frame = pl.read_parquet(path)
                frame.with_columns((1 - pl.col("target")).alias("target")).write_parquet(path)
            other = root / "second"
            build_features(prepared, other, train_parts=1, train_start=1)
            for split, name in [("train", "chosen"), ("val", "validation"), ("test", "evaluation")]:
                self.assertTrue(
                    pl.read_parquet(out / split / f"{name}.parquet").equals(
                        pl.read_parquet(other / split / f"{name}.parquet")
                    )
                )
