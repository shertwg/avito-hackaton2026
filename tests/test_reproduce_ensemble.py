import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import polars as pl

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
