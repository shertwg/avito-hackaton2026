import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import polars as pl
from src.common.click_item2item import build_features, NAMES


def fixture(root):
    m = {"complete": True, "splits": {s: {"parts": []} for s in ("train", "val", "test")}}
    for name, s, observations in [
        ("selected", "train", [(1, 30, 9, [10, 11], [20, 21], [1, 0])]),
        ("donor", "train", [(2, 10, 5, [10, 11], [20, 10], [1, 1]), (3, 20, 6, [10], [20, 21], [0, 1])]),
        ("v", "val", [(4, 40, 9, [10, 11], [20, 21], [0, 1]), (5, 40, 9, [], [20], [0])]),
        ("t", "test", [(6, 50, 9, [10, 11], [20, 21], None)]),
    ]:
        c = []
        f = []
        for feed, ts, u, h, items, y in observations:
            f.append(
                dict(
                    feed_id=feed,
                    timestamp=ts,
                    user_id=u,
                    history__item_id=h,
                    history__category_id=[3] * len(h),
                    history__event_type=[2, 10][: len(h)],
                    history__event_source=[1] * len(h),
                )
            )
            for j, item in enumerate(items):
                r = dict(
                    split=s,
                    feed_id=feed,
                    candidate_index=j,
                    item_id=item,
                    user_id=u,
                    timestamp=ts,
                    category_id=4,
                )
                if y is not None:
                    r.update(target=y[j], pos=j)
                c.append(r)
        cp = f"{name}-c.parquet"
        fp = f"{name}-f.parquet"
        pl.DataFrame(c).with_columns(
            pl.col("feed_id", "item_id").cast(pl.Int32), pl.col("candidate_index").cast(pl.UInt32)
        ).write_parquet(root / cp)
        pl.DataFrame(f).write_parquet(root / fp)
        m["splits"][s]["parts"].append(
            dict(source=cp, feed_path=fp, candidate_path=cp, feeds=len(f), candidates=len(c))
        )
    (root / "manifest.json").write_text(json.dumps(m))
    return m


class ClickGraphTests(unittest.TestCase):
    def test_formulas_and_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp, patch("src.common.click_item2item.BUCKETS", 2):
            root = Path(tmp)
            inp = root / "input"
            inp.mkdir()
            fixture(inp)
            out = root / "features"
            m = build_features(inp, out, train_parts=1, train_start=0)
            self.assertEqual(m["excluded_train_sources"], ["selected-c.parquet"])
            a = pl.read_parquet(out / "val/v-c.parquet")
            p0 = (1 + 20 * 0.75) / 22
            cond = (1 + 12 * p0) / 14
            expected = (
                np.log((cond + 1e-6) / (p0 + 1e-6)) + np.log(((1 + 12 * p0) / 13 + 1e-6) / (p0 + 1e-6))
            ) / 2
            self.assertAlmostEqual(a["ci2i_lift_mean_10"][0], expected, places=6)
            self.assertAlmostEqual(a["coclick_max_10"][0], 1 / np.log2(4), places=6)
            self.assertEqual(a.select(NAMES).row(2), (0.0,) * 18)
            from scripts.audit_click_item2item_features import audit

            checked = audit(inp, out, root / "audit.json")
            self.assertEqual(checked["features"], 18)
            old = (out / "val/v-c.parquet").read_bytes()
            build_features(inp, out, train_parts=1, train_start=0, resume=True)
            self.assertEqual(old, (out / "val/v-c.parquet").read_bytes())

    def test_nonzero_block_excludes_all_selected_labels(self):
        with tempfile.TemporaryDirectory() as tmp, patch("src.common.click_item2item.BUCKETS", 2):
            root = Path(tmp)
            inp = root / "source"
            inp.mkdir()
            fixture(inp)
            a = build_features(inp, root / "first", train_parts=1, train_start=1)
            self.assertEqual(a["excluded_train_sources"], ["donor-c.parquet"])
            for filename in ("donor-c.parquet", "v-c.parquet"):
                path = inp / filename
                df = pl.read_parquet(path)
                df.with_columns((1 - pl.col("target")).alias("target")).write_parquet(path)
            build_features(inp, root / "second", train_parts=1, train_start=1)
            for split, filename in [
                ("train", "donor-c.parquet"),
                ("val", "v-c.parquet"),
                ("test", "t-c.parquet"),
            ]:
                self.assertTrue(
                    pl.read_parquet(root / "first" / split / filename).equals(
                        pl.read_parquet(root / "second" / split / filename)
                    )
                )
