"""Measure domain shift with one deterministic candidate per feed and user-separated OOF."""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.stats import rankdata

from scripts.run_joint_oof import digest
from src.common.preparation import fingerprint, write_json
from src.common.train_selection import select_parts

KEY = ["feed_id", "candidate_index", "item_id"]
BASE = ["f1", "f2", "f3", "f4", "f5", "category_id"]
GROUPS = (
    "cosine",
    "infeed",
    "history",
    "itemstats-008",
    "item2item-008",
    "weighted-item2item-008",
    "category-cooc-008",
    "time-008",
    "event-cosine",
    "feature-ranks-008",
    "userstats-008",
    "recency-cosine",
    "bayesian-ctr-008",
    "feed-context-008",
)


def sample_part(prepared, entry, split, caches):
    candidates = pl.read_parquet(prepared / entry["candidate_path"], columns=[*KEY, "user_id", *BASE])
    # Hash the occurrence, independent of target, exposure position and model predictions.
    sample = (
        candidates.with_columns(pl.struct(["feed_id", "candidate_index"]).hash(seed=42).alias("priority"))
        .sort(["feed_id", "priority", "candidate_index"])
        .unique("feed_id", keep="first", maintain_order=True)
        .drop("priority")
    )
    if sample.height != entry["feeds"]:
        raise ValueError("feed count differs from prepared manifest")
    for root, meta in caches:
        part = meta["splits"][split]["parts"][entry["source"]]
        path = root / part["path"]
        if fingerprint(path) != part["fingerprint"]:
            raise ValueError(f"cache part changed: {path}")
        fields = meta["features"]
        features = pl.read_parquet(path, columns=[*KEY, *fields]).rename(
            {c: f"{root.name}/{c}" for c in fields}
        )
        sample = sample.join(features, on=KEY, how="left", validate="1:1")
    if sample.null_count().to_numpy().any():
        raise ValueError("missing candidate feature values")
    return sample.drop("candidate_index", "item_id")


def auc_score(y, score):
    positive = y == 1
    n = int(positive.sum())
    if n == 0 or n == len(y) or not np.isfinite(score).all():
        raise ValueError("both domains and finite predictions are required")
    return float((rankdata(score)[positive].sum() - n * (n + 1) / 2) / (n * (len(y) - n)))


def classify(source, test, features, trees=300):
    rows = pl.concat([source, test])
    x = rows.select(features).to_numpy().astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError("non-finite features")
    y = np.r_[np.zeros(len(source), dtype=np.int8), np.ones(len(test), dtype=np.int8)]
    user = rows["user_id"].to_numpy()
    if rows["feed_id"].n_unique() != rows.height:
        raise ValueError("overlapping or repeated feeds across domains")
    predictions = np.full(len(y), np.nan)
    gain = np.zeros(len(features))
    splits = []
    folds = rows["user_id"].hash(seed=42).to_numpy() % 4
    for k in range(4):
        train, held = folds != k, folds == k
        if np.intersect1d(user[train], user[held]).size:
            raise ValueError("users overlap across classifier folds")
        counts = np.bincount(y[train], minlength=2)
        weights = train.sum() / (2 * counts[y[train]])
        dataset = lgb.Dataset(x[train], label=y[train], weight=weights)
        model = lgb.train(
            {
                "objective": "binary",
                "num_leaves": 15,
                "min_data_in_leaf": 80,
                "learning_rate": 0.05,
                "num_threads": 2,
                "seed": 42,
                "verbosity": -1,
                "deterministic": True,
                "force_col_wise": True,
            },
            dataset,
            num_boost_round=trees,
        )
        predictions[held] = model.predict(x[held], num_threads=2)
        gain += model.feature_importance(importance_type="gain")
        splits.append(float(auc_score(y[held], predictions[held])))
    auc = float(auc_score(y, predictions))
    ranked = np.argsort(-gain, kind="stable")[:25]
    means = [x[y == k].mean(axis=0, dtype=np.float64) for k in (0, 1)]
    return {
        "source_feeds": len(source),
        "test_feeds": len(test),
        "users": len(np.unique(user)),
        "auc": auc,
        "fold_auc": splits,
        "top_by_gain": [
            {
                "feature": features[i],
                "gain_share": float(gain[i] / max(gain.sum(), 1e-12)),
                "source_mean": float(means[0][i]),
                "test_mean": float(means[1][i]),
            }
            for i in ranked
        ],
    }


def audit(prepared, feature_root, extra_groups=()):
    manifest_path = prepared / "manifest.json"
    m = json.loads(manifest_path.read_bytes())
    selected = select_parts(m, 8)
    names = {e["source"] for e in selected}
    caches = []
    hashes = {}
    groups = (*GROUPS, *extra_groups)
    if len(set(groups)) != len(groups):
        raise ValueError("cache groups must be unique")
    for name in groups:
        root = feature_root / name
        path = root / "manifest.json"
        meta = json.loads(path.read_bytes())
        if not meta["complete"] or (
            "excluded_train_sources" in meta and not names <= set(meta["excluded_train_sources"])
        ):
            raise ValueError(f"incompatible cache: {name}")
        caches.append((root, meta))
        hashes[name] = digest(path)
    data = {}
    for split in ("train", "val", "test"):
        entries = selected if split == "train" else m["splits"][split]["parts"]
        data[split] = pl.concat([sample_part(prepared, e, split, caches) for e in entries])
        print(split, data[split].height, "feeds", flush=True)
    features = [c for c in data["train"].columns if c not in ("feed_id", "user_id")]
    date_proxies = [
        name for name in features if name.startswith("time-008/") or name.endswith("/bc_donor_age_days")
    ]
    without_dates = [name for name in features if name not in date_proxies]
    joint_groups = (
        "cosine/",
        "infeed/",
        "history/",
        "itemstats-008/",
        "item2item-008/",
        "weighted-item2item-008/",
    )
    joint = [name for name in features if name in BASE or name.startswith(joint_groups)]
    report = {
        "prepared_sha256": digest(manifest_path),
        "cache_sha256": hashes,
        "features": len(features),
        "sampling": "one deterministic occurrence per feed; class-balanced fitting; users held separate",
        "trees": 300,
        "seed": 42,
        "excluded_inputs": [
            "target",
            "pos",
            "split",
            "timestamp",
            "item_id",
            "user_id",
            "feed_id",
            "candidate_index",
        ],
        "contrasts": {split: classify(data[split], data["test"], features) for split in ("train", "val")},
        "date_proxy_exclusions": date_proxies,
        "without_date_proxies": {
            split: classify(data[split], data["test"], without_dates) for split in ("train", "val")
        },
        "joint_121": {split: classify(data[split], data["test"], joint) for split in ("train", "val")},
    }
    if extra_groups:
        extra_prefixes = tuple(f"{group}/" for group in extra_groups)
        added = [name for name in features if name.startswith(extra_prefixes)]
        report["added_groups"] = list(extra_groups)
        report["added_feature_count"] = len(added)
        report["joint_121_with_added_groups"] = {
            split: classify(data[split], data["test"], joint + added) for split in ("train", "val")
        }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=Path("data/prepared"))
    parser.add_argument("--features", type=Path, default=Path("data/features"))
    parser.add_argument("--include-group", action="append", default=[], help="append a cache directory name")
    parser.add_argument("--out", type=Path, default=Path("runs/covariate_shift_report.json"))
    args = parser.parse_args()
    report = audit(args.prepared, args.features, args.include_group)
    write_json(args.out, report)
    print({s: v["auc"] for s, v in report["contrasts"].items()})
