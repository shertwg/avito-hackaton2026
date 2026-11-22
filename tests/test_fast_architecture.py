"""Регрессионные тесты ускоренной архитектуры: FastNDCGEvaluator, _safe_zscore, hybrid_rankgauss_025 и раннер GBDT."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import polars as pl

from scripts.reproduce_ensemble import rebuild_ensemble, sha256
from src.common.metric import FastNDCGEvaluator, ndcg_one
from src.common.scores import feed_hybrid_rankgauss
from src.gbdt.feed_interactions import add_feed_interactions
from src.gbdt.run_exp import get_args
from src.gbdt.weights import ipw_weights, position_ctr


class FastArchitectureTests(unittest.TestCase):
    def test_fast_ndcg_matches_reference_ndcg_one_on_all_edge_cases(self):
        rng = np.random.default_rng(2026)
        feed_ids, targets, scores = [], [], []
        sizes = [1, 2, 5, 10, 29, 30, 31, 64, 128, 129, 250] * 4
        for idx, sz in enumerate(sizes):
            f_id = (idx + 1) * 13
            t = rng.integers(0, 2, size=sz)
            if idx % 5 == 0:
                t[:] = 0
            elif idx % 5 == 1:
                t[:] = 1
            # Много связанных скоров для жёсткой проверки пессимистичного тай-брейка
            s = rng.integers(-3, 4, size=sz).astype(np.float64)
            feed_ids.extend([f_id] * sz)
            targets.extend(t.tolist())
            scores.extend(s.tolist())

        f_arr = np.asarray(feed_ids, dtype=np.int32)
        t_arr = np.asarray(targets, dtype=np.int8)
        s_arr = np.asarray(scores, dtype=np.float64)
        perm = rng.permutation(len(f_arr))
        f_shuf, t_shuf, s_shuf = f_arr[perm], t_arr[perm], s_arr[perm]

        for cutoff, idcg_k in ((30, None), (10, None), (None, None), (None, 30)):
            ev = FastNDCGEvaluator(f_shuf, t_shuf, cutoff=cutoff, idcg_k=idcg_k)
            uf, pf = ev.per_feed(s_shuf)
            expected = []
            for fid in uf:
                m = f_shuf == fid
                expected.append(ndcg_one(t_shuf[m], s_shuf[m], cutoff=cutoff, idcg_k=idcg_k))
            np.testing.assert_allclose(pf, np.asarray(expected), rtol=0, atol=1e-14)
            self.assertAlmostEqual(ev.mean(s_shuf), float(np.mean(expected)), delta=1e-14)

    def test_safe_zscore_zeroes_constant_float32_groups(self):
        # 0.333199 в Float32 давало ненулевой шум (col - mean) / (std + 1e-6) в старом коде
        df = pl.DataFrame(
            {
                "feed_id": [1, 1, 1, 2, 2],
                "category_id": [7, 7, 7, 8, 8],
                "f1": [0.333199, 0.333199, 0.333199, 0.1, 0.9],
                "f5": [0.415174, 0.415174, 0.415174, 0.2, 0.2],
                "is_item_ctr": [0.0213, 0.0213, 0.0213, 0.01, 0.05],
                "is_item_position_ratio": [1.0, 1.0, 1.0, 0.8, 1.4],
                "svd_cos_max_50": [0.0, 0.0, 0.0, 0.0, 0.6],
                "cos_user": [0.2, 0.5, 0.8, 0.1, 0.4],
            }
        )
        out, cols = add_feed_interactions(df)
        self.assertIn("fi_is_item_position_ratio_z", cols)
        self.assertIn("fi_svd_cos_max_50_z", cols)
        self.assertIn("fi_svd50_x_cos", cols)
        f1_feed1 = out.filter(pl.col("feed_id") == 1)
        np.testing.assert_array_equal(f1_feed1["fi_incat_f1_z"].to_numpy(), np.zeros(3, dtype=np.float32))
        np.testing.assert_array_equal(f1_feed1["fi_incat_f5_z"].to_numpy(), np.zeros(3, dtype=np.float32))
        np.testing.assert_array_equal(f1_feed1["fi_is_item_ctr_z"].to_numpy(), np.zeros(3, dtype=np.float32))

    def test_reproduce_ensemble_supports_hybrid_rankgauss_025(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            val_dir = root / "val" / "impressions"
            val_dir.mkdir(parents=True)
            pl.DataFrame(
                {
                    "feed_id": [1, 2],
                    "timestamp": [10, 20],
                    "item_id": [[1, 2, 3], [4, 5]],
                    "target": [[0, 1, 0], [1, 0]],
                }
            ).write_parquet(val_dir / "part-00000.parquet")
            keys = pl.DataFrame({"feed_id": [1, 1, 1, 2, 2], "item_id": [1, 2, 3, 4, 5]}).cast(pl.Int32)
            keys.with_columns(pl.lit(0.0, dtype=pl.Float32).alias("score")).write_parquet(
                root / "sample_submission.parquet"
            )
            comp = root / "m1"
            comp.mkdir()
            pred = keys.with_columns(pl.Series("score", [0.1, 0.9, 0.3, 2.0, 1.0], dtype=pl.Float32))
            hashes = {}
            for s in ("val", "test"):
                p = comp / f"{s}_pred.parquet"
                pred.write_parquet(p)
                hashes[s] = sha256(p)
            recipe = root / "recipe.json"
            recipe.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "members": [
                            {
                                "path": "m1",
                                "weight": 1.0,
                                "normalization": "hybrid_rankgauss_025",
                                "sha256": hashes,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            rep = rebuild_ensemble(root, root / "out", recipe)
            self.assertEqual(rep["submission_validator"], "passed")
            actual = pl.read_parquet(root / "out" / "test_pred.parquet")["score"].to_numpy()
            expected = feed_hybrid_rankgauss(keys["feed_id"].to_numpy(), pred["score"].to_numpy(), rank_alpha=0.25)
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_pointwise_zero_grad_guard_and_ipw_all_normalization(self):
        with self.assertRaises(SystemExit):
            get_args(["--data", "competition", "--out", "runs/x", "--kind", "lightgbm", "--loss", "binary", "--drop-zero-grad-feeds"])
        pos = np.array([0, 1, 2, 10, 20, 20] * 40, dtype=np.float32)
        tgt = np.array([1, 0, 1, 0, 1, 0] * 40, dtype=np.int8)
        ctr = position_ctr(pos, tgt, min_count=5)
        w_all = ipw_weights(pos, tgt, ctr, max_w=5.0, positives_only=False, normalize_mean=True)
        self.assertAlmostEqual(float(w_all.mean()), 1.0, places=5)
        # При одинаковой позиции (pos=20 на индексах 4 и 5) позитив и негатив получают одинаковый вес 1/p(pos)
        self.assertAlmostEqual(float(w_all[4]), float(w_all[5]), places=5)


if __name__ == "__main__":
    unittest.main()
