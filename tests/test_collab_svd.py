import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl

from scripts.audit_collab_svd_features import audit
from src.common.collab_svd import (
    ARRAYS,
    NAMES,
    build_features,
    factorize,
    score_part,
    shifted_ppmi,
)


def fixture(root):
    groups = {
        "train": {
            "donor": [(i, i, [1, 2] if i < 5 else [3, 4] if i < 9 else [5, 6]) for i in range(1, 13)]
            + [(13, 1, [1, 2])],
            "selected": [(101, 1, [1, 99])],
        },
        "val": {"val": [(201, 10, [1, 3, 99])]},
        "test": {"test": [(301, 11, [2, 4, 5, 6, 99])]},
    }
    manifest = {"complete": True, "splits": {}}
    for split, shards in groups.items():
        parts = []
        for shard, rows in shards.items():
            records = []
            feeds = []
            for feed, user, items in rows:
                feeds.append((feed, [1, 3], [2, 1]))
                for offset, item in enumerate(items):
                    row = {
                        "split": split,
                        "feed_id": feed,
                        "user_id": user,
                        "candidate_index": offset,
                        "item_id": item,
                    }
                    if split != "test":
                        row["target"] = 1
                    records.append(row)
            candidates = pl.DataFrame(records).with_columns(
                pl.col("feed_id", "item_id").cast(pl.Int32), pl.col("candidate_index").cast(pl.UInt32)
            )
            candidates.write_parquet(root / f"{shard}.parquet")
            pl.DataFrame(
                feeds, schema=["feed_id", "history__item_id", "history__event_type"], orient="row"
            ).write_parquet(root / f"{shard}-feeds.parquet")
            parts.append(
                {
                    "source": f"{shard}.parquet",
                    "candidate_path": f"{shard}.parquet",
                    "feed_path": f"{shard}-feeds.parquet",
                    "candidates": len(candidates),
                    "feeds": len(feeds),
                }
            )
        manifest["splits"][split] = {"parts": parts}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


class CollaborativeFeaturesTests(unittest.TestCase):
    def test_two_hop_factorization_and_held_shard_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            fixture(source)
            cache = root / "first"
            options = {"train_parts": 1, "train_start": 1, "min_degree": 2, "rank": 3}
            report = build_features(source, cache, **options)
            self.assertEqual(report["excluded_train_sources"], ["selected.parquet"])
            self.assertEqual(report["features"], NAMES)
            checked = audit(source, cache, root / "audit.json")
            self.assertEqual(checked["graph"]["donor_distinct_click_edges"], 24)
            vectors = np.load(cache / ARRAYS[1])
            self.assertAlmostEqual(float(vectors[0] @ vectors[1]), 1.0, places=6)
            self.assertAlmostEqual(float(vectors[0] @ vectors[2]), 0.0, places=6)
            before = (cache / "val/val.parquet").read_bytes()
            build_features(source, cache, resume=True, **options)
            self.assertEqual(before, (cache / "val/val.parquet").read_bytes())
            for name in ("selected", "val"):
                path = source / f"{name}.parquet"
                pl.read_parquet(path).with_columns(pl.lit(0).alias("target")).write_parquet(path)
            after = root / "second"
            build_features(source, after, **options)
            for filename in ("train/selected.parquet", "val/val.parquet", "test/test.parquet"):
                self.assertTrue(pl.read_parquet(cache / filename).equals(pl.read_parquet(after / filename)))

    def test_shifted_pmi_and_position_masks(self):
        edges = {"item_id": [1, 2, 3, 4], "user_id": [1, 2, 3, 4]}
        _, graph = shifted_ppmi(edges)
        np.testing.assert_allclose(graph.toarray(), np.eye(4) * np.log(2), rtol=1e-7)
        _, singular = factorize(graph, 4)
        np.testing.assert_allclose(singular, np.full(4, np.log(2)), rtol=1e-6)
        candidates = pl.DataFrame(
            {"split": ["val"] * 2, "feed_id": [1, 1], "candidate_index": [0, 1], "item_id": [1, 0]}
        )
        feeds = pl.DataFrame(
            {"feed_id": [1], "history__item_id": [[0] * 10 + [2]], "history__event_type": [[1] * 10 + [2]]}
        )
        result = score_part(candidates, feeds, np.array([1, 2]), np.array([[1.0], [-1.0]], dtype=np.float32))
        np.testing.assert_allclose(result.select(NAMES).row(0), [1, 0, 0, 0, -1, -0.5, -1, -0.5])
        np.testing.assert_allclose(result.select(NAMES).row(1), np.zeros(8))
