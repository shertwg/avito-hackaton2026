"""Rebuild a frozen ensemble from hashed component predictions."""

import hashlib
import json
from math import fsum, isclose, isfinite
from pathlib import Path
from re import fullmatch
from tempfile import TemporaryDirectory

import numpy as np
import polars as pl
from scipy.special import ndtri

from scripts.compare_team_overlay import KEY, aligned
from scripts.validate_submission import validate
from src.common.cv import pseudo_split
from src.common.data import load_labels
from src.common.metric import per_feed_ndcg
from src.common.preparation import write_json
from src.common.scores import feed_rank01, feed_zscore

SPLITS = ("val", "test")


def hybrid_rankgauss_025(feed, score):
    gaussian = ndtri(np.clip(feed_rank01(feed, score), 1e-4, 1.0 - 1e-4))
    return (0.75 * feed_zscore(feed, score) + 0.25 * feed_zscore(feed, gaussian)).astype(np.float32)


NORMALIZERS = {
    "zscore": feed_zscore,
    "rank01": feed_rank01,
    "raw": lambda _, score: score,
    "hybrid_rankgauss_025": hybrid_rankgauss_025,
}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_config(path):
    contents = path.read_bytes()
    config = json.loads(contents)
    if config.get("version") != 1 or not config.get("members"):
        raise ValueError("ensemble recipe version 1 requires at least one member")
    aggregation = config.get("aggregation", "weighted_sum_float64")
    row_order = config.get("row_order", "feed_id_item_id")
    if aggregation not in ("weighted_sum_float64", "mean_float32"):
        raise ValueError(f"unknown ensemble aggregation: {aggregation}")
    if row_order not in ("feed_id_item_id", "raw_feed_stable"):
        raise ValueError(f"unknown ensemble row order: {row_order}")
    expected_outputs = config.get("expected_output_sha256", {})
    if any(
        split not in SPLITS or not fullmatch(r"[0-9a-f]{64}", digest)
        for split, digest in expected_outputs.items()
    ):
        raise ValueError("expected_output_sha256 must map val/test to SHA-256 strings")
    members = []
    for entry in config["members"]:
        root = Path(entry["path"])
        if root.is_absolute() or not entry["path"]:
            raise ValueError("component paths must be relative to the recipe")
        weight = entry["weight"]
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not isfinite(weight)
            or weight < 0
        ):
            raise ValueError("weights must be finite, nonnegative numbers")
        normalization = entry["normalization"]
        if normalization not in NORMALIZERS:
            raise ValueError(f"unknown score normalization: {normalization}")
        paths = {split: (path.parent / root / f"{split}_pred.parquet").resolve() for split in SPLITS}
        if any(paths == existing["paths"] for existing in members):
            raise ValueError(f"duplicate ensemble member: {root}")
        for split in SPLITS:
            expected = entry["sha256"][split]
            if not fullmatch(r"[0-9a-f]{64}", expected):
                raise ValueError(f"invalid input SHA-256 for {paths[split]}")
            if sha256(paths[split]) != expected:
                raise ValueError(f"input SHA-256 differs: {paths[split]}")
        members.append({**entry, "paths": paths})
    if not isclose(fsum(m["weight"] for m in members), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("frozen ensemble weights must sum to one")
    if aggregation == "mean_float32" and any(m["weight"] != 1 / len(members) for m in members):
        raise ValueError("mean_float32 requires equal weights of 1/member_count")
    return members, aggregation, row_order, expected_outputs, hashlib.sha256(contents).hexdigest()


def rebuild_ensemble(raw, output, recipe):
    raw, output, recipe = (Path(p).resolve() for p in (raw, output, recipe))
    if output.exists():
        raise FileExistsError("choose a fresh output directory for the ensemble")
    if not (raw / "sample_submission.parquet").exists() and (raw / "competition").is_dir():
        raw = raw / "competition"
    template = raw / "sample_submission.parquet"
    template_digest = sha256(template)
    members, aggregation, row_order, expected_outputs, recipe_digest = read_config(recipe)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "recipe_sha256": recipe_digest,
        "template_sha256": template_digest,
        "aggregation": aggregation,
        "row_order": row_order,
        "expected_output_sha256": expected_outputs,
        "members": [
            {key: member[key] for key in ("path", "weight", "normalization", "sha256")} for member in members
        ],
        "predictions": {},
    }
    with TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / "ensemble"
        stage.mkdir()
        for split in SPLITS:
            if split == "val" or row_order == "raw_feed_stable":
                source = load_labels(raw, split)
            else:
                source = pl.read_parquet(template, columns=KEY)
            source = source.with_columns(pl.col(KEY).cast(pl.Int32))
            if row_order == "raw_feed_stable":
                source = source.sort("feed_id", maintain_order=True).with_row_index("_raw_row")
            source = source.sort(KEY)
            if source.select(KEY).is_duplicated().any() or source.null_count().to_numpy().any():
                raise ValueError(f"duplicate keys or nulls in {split} reference")
            order = (
                source["_raw_row"].arg_sort().to_numpy() if row_order == "raw_feed_stable" else slice(None)
            )
            ordered = source[order]
            feed = ordered["feed_id"].to_numpy()
            if aggregation == "mean_float32":
                matrix = np.empty((len(feed), len(members)), dtype=np.float32)
            else:
                score = np.zeros(len(feed), dtype=np.float64)
            for column, member in enumerate(members):
                raw_score = aligned(source, member["paths"][split])[order]
                transform = NORMALIZERS[member["normalization"]]
                values = transform(feed, raw_score.astype(np.float64, copy=False))
                if aggregation == "mean_float32":
                    matrix[:, column] = values
                else:
                    score += member["weight"] * values
            if aggregation == "mean_float32":
                score = matrix.mean(axis=1)
                del matrix
            score = score.astype(np.float32)
            if not np.isfinite(score).all():
                raise ValueError(f"non-finite ensemble scores: {split}")
            frame = ordered.select(KEY).with_columns(pl.Series("score", score))
            if split == "val":
                frame = frame.with_columns(ordered["target"])
                pub, priv = pseudo_split(feed, ordered["timestamp"])
                ids, ndcg = per_feed_ndcg(feed, ordered["target"].to_numpy(), score, cutoff=30)
                report["ndcg30"] = {
                    name: float(ndcg[np.isin(ids, subset)].mean()) if len(subset) else None
                    for name, subset in (("pseudo_public", pub), ("pseudo_private", priv), ("all", ids))
                }
            path = stage / f"{split}_pred.parquet"
            frame.write_parquet(path)
            digest = sha256(path)
            if split in expected_outputs and digest != expected_outputs[split]:
                raise ValueError(
                    f"{path.name} SHA-256 differs: expected {expected_outputs[split]}, got {digest}"
                )
            report["predictions"][path.name] = {"rows": len(frame), "sha256": digest}
        errors = validate(str(template), str(stage / "test_pred.parquet"))
        if errors:
            raise ValueError(errors)
        for member in members:
            for split in SPLITS:
                if sha256(member["paths"][split]) != member["sha256"][split]:
                    raise ValueError(
                        f"input changed during ensemble reconstruction: {member['paths'][split]}"
                    )
        if sha256(recipe) != recipe_digest or sha256(template) != template_digest:
            raise ValueError("recipe or submission template changed during ensemble reconstruction")
        report.update(complete=True, submission_validator="passed")
        write_json(stage / "pipeline.json", report)
        stage.rename(output)
    return report
