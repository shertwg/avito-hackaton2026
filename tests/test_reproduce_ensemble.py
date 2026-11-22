import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import polars as pl
from scipy.special import ndtri

from scripts.reproduce_ensemble import rebuild_ensemble, sha256
from scripts.reproduce_ranking import main


class EnsembleReplayTests(unittest.TestCase):
    def test_replays_locked_scores_and_rejects_changed_artifacts(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            validation = root / "val" / "impressions"
            validation.mkdir(parents=True)
            pl.DataFrame(
                {
                    "feed_id": [1, 2, 3],
                    "timestamp": [10, 20, 30],
                    "item_id": [[1, 2], [1, 2], [5]],
                    "target": [[0, 1], [1, 0], [1]],
                }
            ).write_parquet(validation / "part-00000.parquet")
            keys = pl.DataFrame({"feed_id": [1, 1, 2, 2, 3], "item_id": [1, 2, 1, 2, 5]}).cast(pl.Int32)
            keys.with_columns(pl.lit(0.0, dtype=pl.Float32).alias("score")).write_parquet(
                root / "sample_submission.parquet"
            )
            recipe = root / "frozen.json"
            members = []
            for name, scores, weight, normalization in (
                ("base", [0, 2, 4, 2, 9], 0.75, "zscore"),
                ("extra", [2, 0, 0, 4, 7], 0.25, "rank01"),
            ):
                component = root / name
                component.mkdir()
                prediction = keys.with_columns(pl.Series("score", scores, dtype=pl.Float32)).reverse()
                hashes = {}
                for split in ("val", "test"):
                    path = component / f"{split}_pred.parquet"
                    prediction.write_parquet(path)
                    hashes[split] = sha256(path)
                members.append(
                    {"path": name, "weight": weight, "normalization": normalization, "sha256": hashes}
                )
            config = {"version": 1, "members": members}
            recipe.write_text(json.dumps(config), encoding="utf-8")
            output = root / "replayed"
            argv = [
                "reproduce_ranking",
                "--raw",
                str(root),
                "--out",
                str(output),
                "--ensemble-config",
                str(recipe),
            ]
            with (
                patch.object(sys, "argv", argv),
                patch("scripts.reproduce_ranking.reproduce_ranking") as train,
            ):
                main()
                train.assert_not_called()
            replay = pl.read_parquet(output / "test_pred.parquet")
            self.assertTrue(replay.select(keys.columns).equals(keys))
            z = 1 / (np.sqrt(2) + 1e-9)
            np.testing.assert_allclose(
                replay["score"].to_numpy(),
                [0.25 - 0.75 * z, 0.125 + 0.75 * z, 0.125 + 0.75 * z, 0.25 - 0.75 * z, 0.25],
            )
            report = json.loads((output / "pipeline.json").read_bytes())
            self.assertEqual(report["submission_validator"], "passed")
            self.assertEqual(
                report["predictions"]["test_pred.parquet"]["sha256"], sha256(output / "test_pred.parquet")
            )
            altered = root / "extra" / "test_pred.parquet"
            prediction = pl.read_parquet(altered)
            pl.concat([prediction.slice(1, 1), prediction.slice(1)]).write_parquet(altered)
            with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
                rebuild_ensemble(root, root / "tampered", recipe)
            self.assertFalse((root / "tampered").exists())
            members[1]["sha256"]["test"] = sha256(altered)
            recipe.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "keys differ"):
                rebuild_ensemble(root, root / "duplicates", recipe)
            self.assertFalse((root / "duplicates").exists())

    def test_18_member_hybrid_uses_float32_mean_and_checks_output_digest(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = pl.DataFrame(
                {
                    "feed_id": [2, 1, 3],
                    "timestamp": [20, 10, 30],
                    "item_id": [[4, 1], [9, 2, 7], [8]],
                    "target": [[0, 1], [1, 0, 0], [1]],
                }
            )
            for split in ("val", "test"):
                folder = root / split / "impressions"
                folder.mkdir(parents=True)
                (raw if split == "val" else raw.drop("target")).write_parquet(folder / "part-00000.parquet")
            keys = (
                raw.select("feed_id", "item_id")
                .explode("item_id")
                .cast(pl.Int32)
                .sort("feed_id", maintain_order=True)
            )
            keys.reverse().write_parquet(root / "sample_submission.parquet")

            def standardize(x):
                return (x - x.mean()) / (x.std(ddof=1) + 1e-9) if len(x) > 1 else np.zeros_like(x)

            rng = np.random.default_rng(42)
            members, columns = [], []
            feed_ids = keys["feed_id"].to_numpy()
            for i in range(18):
                scores = rng.normal(size=len(keys))
                scores[1] = scores[0] if i % 2 else scores[1]
                component = root / f"model-{i}"
                component.mkdir()
                hashes = {}
                for split in ("val", "test"):
                    path = component / f"{split}_pred.parquet"
                    keys.with_columns(pl.Series("score", scores)).reverse().write_parquet(path)
                    hashes[split] = sha256(path)
                members.append(
                    {
                        "path": component.name,
                        "weight": 1 / 18,
                        "normalization": "hybrid_rankgauss_025",
                        "sha256": hashes,
                    }
                )
                parts = []
                for feed in (1, 2, 3):
                    vec = scores[feed_ids == feed]
                    ranks = np.array(
                        [(vec < value).sum() + ((vec == value).sum() + 1) / 2 for value in vec]
                    ) / len(vec)
                    gauss = ndtri(np.clip(ranks, 1e-4, 1 - 1e-4))
                    parts.append((0.75 * standardize(vec) + 0.25 * standardize(gauss)).astype(np.float32))
                columns.append(np.concatenate(parts))
            expected = root / "expected.parquet"
            keys.with_columns(pl.Series("score", np.column_stack(columns).mean(axis=1))).write_parquet(
                expected
            )
            config = {
                "version": 1,
                "members": members,
                "aggregation": "mean_float32",
                "row_order": "raw_feed_stable",
                "expected_output_sha256": {"test": sha256(expected)},
            }
            recipe = root / "frozen-hybrid.json"
            recipe.write_text(json.dumps(config), encoding="utf-8")
            output = root / "replayed-hybrid"
            report = rebuild_ensemble(root, output, recipe)
            self.assertEqual(report["predictions"]["test_pred.parquet"]["sha256"], sha256(expected))
            self.assertEqual((output / "test_pred.parquet").read_bytes(), expected.read_bytes())
            config["expected_output_sha256"]["test"] = "0" * 64
            recipe.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "test_pred.parquet SHA-256 differs"):
                rebuild_ensemble(root, root / "mismatched", recipe)
            self.assertFalse((root / "mismatched").exists())
