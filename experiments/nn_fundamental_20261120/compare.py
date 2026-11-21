"""Reproduce paired OOF and forward comparisons for the November 20 audit."""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from scripts.compare_team_overlay import WEIGHTS
from src.common.cv import pseudo_split
from src.common.data import load_labels
from src.common.metric import per_feed_ndcg
from src.common.scores import feed_zscore

KEY = ["feed_id", "item_id"]
RUNS = Path("runs")
OUT = RUNS / "fundamental-audit-001"
FINALISTS = ("fusion2", "balanced")
CONTROLS = {
    "fusion": ["periodic42"],
    "fusion43": ["periodic43"],
    "two_seed": ["rule42"],
    "fusion2": ["periodic2", "two_seed", "rule42"],
    "balanced": ["two_seed", "fusion2", "rule42"],
    "overlay_fusion2": ["overlay_two_seed", "overlay_rule42"],
    "overlay_balanced": ["overlay_two_seed", "overlay_rule42"],
}


def paths(forward):
    prefix = "nn-v2-forward" if forward else "nn-v2"
    folders = {
        f"{kind}{seed}": RUNS / f"{prefix}-{kind}-s{seed}-001"
        for kind in ("control", "periodic") for seed in (42, 43)
    }
    if not forward:
        folders["control43"] = RUNS / "nn-causal-order-time-s43-002"
    prefix = "nn-forward-context" if forward else "nn-context"
    folders.update(fusion=RUNS / f"{prefix}-fusion-s42-001", fusion43=RUNS / f"{prefix}-fusion-s43-001")
    return folders


def attach(reference, path, name):
    pred = pl.read_parquet(path).select(*KEY, pl.col("score").alias(name))
    if pred.height != reference.height:
        raise ValueError(f"Row count mismatch for {name}")
    result = reference.join(pred, on=KEY, how="left", validate="1:1")
    if result[name].null_count() or not result[name].is_finite().all():
        raise ValueError(f"Missing predictions for {name}")
    return result


def paired_interval(delta):
    rng = np.random.default_rng(42)
    draws = [rng.choice(delta, len(delta)).mean() for _ in range(2000)]
    return [delta.mean(), *np.quantile(draws, [.025, .975])]


def compare(forward, anchor):
    folders = paths(forward)
    ref = load_labels("competition", "val").cast(dict.fromkeys(KEY, pl.Int32))
    folds = pl.read_parquet(folders["control42"] / "folds.parquet").sort("feed_id")
    ref = ref.join(folds.select("feed_id", "fold"), on="feed_id", validate="m:1")
    filename = "forward_pred.parquet" if forward else "val_pred.parquet"
    if forward:
        ref = ref.filter(pl.col("fold") >= 2)
    for name, folder in folders.items():
        if not folds.equals(pl.read_parquet(folder / "folds.parquet").sort("feed_id")):
            raise ValueError(f"Different folds for {name}")
        ref = attach(ref, folder / filename, name)
    if anchor:
        ref = attach(ref, anchor / "val_pred.parquet", "anchor")
    ref = ref.sort(KEY)
    feed, target = (ref[c].to_numpy() for c in ("feed_id", "target"))
    ids, sizes = np.unique(feed, return_counts=True)
    long_rows = np.repeat(sizes >= 151, sizes)
    scores = {key: feed_zscore(feed, ref[key].to_numpy()) for key in folders}
    for seed in (42, 43):
        scores[f"rule{seed}"] = np.where(long_rows, scores[f"control{seed}"], scores[f"periodic{seed}"])
    scores["periodic2"] = (scores["periodic42"] + scores["periodic43"]) / 2
    scores["two_seed"] = (scores["rule42"] + scores["rule43"]) / 2
    scores["fusion2"] = (scores["fusion"] + scores["fusion43"]) / 2
    scores["balanced"] = (scores["fusion2"] + scores["two_seed"]) / 2

    if forward:
        cuts = {f"fold{k}": np.isin(ids, ref.filter(pl.col("fold") == k)["feed_id"].unique()) for k in (2, 3)}
    else:
        pub, priv = pseudo_split(feed, ref["timestamp"].to_numpy())
        cuts = {"pub": np.isin(ids, pub), "priv": np.isin(ids, priv), "all": np.ones(len(ids), bool)}
        cuts.update(priv_long=cuts["priv"] & (sizes >= 151), priv_short=cuts["priv"] & (sizes < 151))
        ref.select(*KEY, *[pl.Series(k, scores[k].astype("float32")) for k in FINALISTS]).write_parquet(
            OUT / "candidates.parquet"
        )
    if anchor:
        base = feed_zscore(feed, ref["anchor"].to_numpy())
        early, late = np.isin(feed, pub), np.isin(feed, priv)
        grids = {}
        selected_quality = {}
        for name in ("rule42", "two_seed", *FINALISTS):
            neural = feed_zscore(feed, scores[name])
            scores["overlay_" + name] = .7 * base + .3 * neural
            def evaluate(mask, weight):
                mixed = (1 - weight) * base[mask] + weight * neural[mask]
                return per_feed_ndcg(feed[mask], target[mask], mixed, cutoff=30)[1].mean()
            grid = [evaluate(early, weight) for weight in WEIGHTS]
            weight = WEIGHTS[int(np.argmax(grid))]
            blended = (1 - weight) * base[late] + weight * neural[late]
            selected_quality[name] = per_feed_ndcg(feed[late], target[late], blended, cutoff=30)[1]
            grids[name] = {"weights": WEIGHTS, "pub_grid": grid, "selected_weight": weight, "private": evaluate(late, weight)}
        for name in FINALISTS:
            grids[name]["vs_public_selected_two_seed"] = paired_interval(selected_quality[name] - selected_quality["two_seed"])
        (OUT / "anchor_weights.json").write_text(json.dumps(grids, indent=2))

    quality = {name: per_feed_ndcg(feed, target, s, cutoff=30)[1] for name, s in scores.items()}
    result = {}
    for label, mask in cuts.items():
        result[label] = {}
        for name, q in quality.items():
            row = {"n": int(mask.sum()), "ndcg": q[mask].mean()}
            for baseline in CONTROLS.get(name, ()):
                row["vs_" + baseline] = paired_interval((q - quality[baseline])[mask])
            result[label][name] = row
        print(label, {k: round(result[label][k]["ndcg"], 5) for k in FINALISTS}, flush=True)
    filename = "forward_replicate.json" if forward else "oof_replicate.json"
    (OUT / filename).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--forward", action="store_true")
    mode.add_argument("--anchor", type=Path, help="Directory containing a full validation prediction file")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    compare(args.forward, args.anchor)
