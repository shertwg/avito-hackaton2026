"""Train a target-free Production Ranker (`pos`) Teacher on disjoint block-032-064 shards.

Builds `data/features/pos-teacher-032` for downstream 32-shard GBDTs and exports
standalone `runs/pos-teacher-032/{val_pred.parquet, test_pred.parquet, metrics.json}`.
Strictly uses only candidate/feed/cosine/infeed/history features + feed interactions;
clicks (`target`) are never read during feature loading or training.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostRanker, Pool

from src.common.cv import mask_feeds, pseudo_split
from src.common.feature_cache import build_cache, cache_lock
from src.common.metric import all_variants
from src.common.preparation import write_json, write_parquet_atomic
from src.common.similarity import attach_features
from src.common.train_selection import select_parts
from src.gbdt.feed_interactions import add_feed_interactions

KEY_COLS = ("split", "feed_id", "candidate_index", "item_id")
BASE_CAND_COLS = (
    "split",
    "feed_id",
    "candidate_index",
    "item_id",
    "f1",
    "f2",
    "f3",
    "f4",
    "f5",
    "category_id",
)
PT_NAMES = [
    "pt_gain",
    "pt_pct",
    "pt_gain_z",
    "pt_gain_rank",
    "pt_gain_dmax",
    "pt_pct_z",
]
PT_DOMAINS = {
    "pt_gain": (None, None),
    "pt_pct": (None, None),
    "pt_gain_z": (None, None),
    "pt_gain_rank": (0, 1),
    "pt_gain_dmax": (None, 0),
    "pt_pct_z": (None, None),
}
POLICY = (
    "Trained strictly on production `pos` over 30 disjoint block-032-064 donor shards; "
    "excluded from downstream 32-shard GBDT training shards; target labels never used."
)


def _load_shard_features(
    prepared_dir: Path,
    entry: dict,
    split: str,
    sources: list[Path],
    *,
    include_pos: bool = False,
) -> pl.DataFrame:
    """Load one prepared shard (target-free) and attach feature sources + feed interactions."""
    cols = list(BASE_CAND_COLS) + (["pos"] if include_pos else [])
    df = (
        pl.read_parquet(prepared_dir / entry["candidate_path"], columns=cols)
        .sort(["feed_id", "candidate_index"])
    )
    for src_root in sources:
        df, _ = attach_features(df, src_root, split, [entry["source"]])
    df, _ = add_feed_interactions(df)
    return df


def _to_pandas_features(df: pl.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Zero-copy dict-backed Pandas DataFrame with int32 category_id and finite float32 cols."""
    cols: dict[str, np.ndarray] = {}
    for col in feature_cols:
        arr = df[col].to_numpy()
        if col == "category_id":
            cols[col] = np.nan_to_num(arr, nan=-1).astype(np.int32, copy=False)
        else:
            f = arr.astype(np.float32, copy=False)
            if not np.isfinite(f).all():
                f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
            cols[col] = f
    return pd.DataFrame(cols, copy=False)


def _find_disjoint_donors(
    prepared_manifest: dict,
    features_dir: Path,
    donor_block: str,
    train_parts: int,
) -> tuple[set[str], list[dict]]:
    """Return 32 excluded target-training sources and disjoint block-032-064 donor entries."""
    chosen = select_parts(prepared_manifest, train_parts)
    excluded = {p["source"] for p in chosen}
    blk_hist_manifest = json.loads(
        (features_dir / donor_block / "history" / "manifest.json").read_text(encoding="utf-8")
    )
    blk_sources = set(blk_hist_manifest["splits"]["train"]["parts"])
    donor_sources = blk_sources - excluded
    if not donor_sources:
        raise ValueError("No disjoint donor shards remain after excluding train parts")
    donor_entries = [
        p for p in prepared_manifest["splits"]["train"]["parts"] if p["source"] in donor_sources
    ]
    if len(donor_entries) != len(donor_sources):
        raise ValueError(
            f"Prepared manifest missing donor entries: {len(donor_entries)} != {len(donor_sources)}"
        )
    return excluded, donor_entries


def _fit_or_load_models(
    prepared_dir: Path,
    features_dir: Path,
    run_dir: Path,
    donor_entries: list[dict],
    donor_block: str,
    donor_cosine: str,
    *,
    resume: bool,
) -> tuple[CatBoostRanker, CatBoostRanker, list[str]]:
    """Train both GPU feed-grouped ranker pos-teacher models on disjoint donors or reload on --resume."""
    run_dir.mkdir(parents=True, exist_ok=True)
    qrmse_path = run_dir / "model_qrmse.cbm"
    yeti_path = run_dir / "model_yeti.cbm"
    cols_path = run_dir / "feature_cols.json"

    if resume and qrmse_path.exists() and yeti_path.exists() and cols_path.exists():
        print(f"pos-teacher: reloading checkpoints from {run_dir}", flush=True)
        feature_cols = json.loads(cols_path.read_text(encoding="utf-8"))
        model_qrmse = CatBoostRanker().load_model(str(qrmse_path))
        model_yeti = CatBoostRanker().load_model(str(yeti_path))
        return model_qrmse, model_yeti, feature_cols

    donor_sources = [
        features_dir / donor_cosine,
        features_dir / "infeed",
        features_dir / donor_block / "history",
    ]
    total_rows = sum(int(entry["candidates"]) for entry in donor_entries)
    print(
        f"pos-teacher: loading {len(donor_entries)} disjoint donor shards ({total_rows:,} rows)...",
        flush=True,
    )

    feature_cols: list[str] | None = None
    cols_buf: dict[str, np.ndarray] = {}
    y_gain = np.empty(total_rows, dtype=np.float32)
    group_ids = np.empty(total_rows, dtype=np.int32)
    offset = 0
    excluded_meta = set(KEY_COLS) | {"pos", "target"}

    for idx, entry in enumerate(donor_entries, 1):
        df = _load_shard_features(prepared_dir, entry, "train", donor_sources, include_pos=True)
        if df["pos"].null_count() > 0 or (df["pos"] < 0).any():
            raise ValueError(f"Unexpected null/negative pos in donor shard {entry['source']}")
        df = df.with_columns(
            (1.0 / (pl.col("pos").cast(pl.Float32) + 2.0).log(2.0)).alias("_y_gain")
        )
        if feature_cols is None:
            feature_cols = sorted(
                c for c in df.columns if c not in excluded_meta and c != "_y_gain"
            )
            cols_buf = {
                c: np.empty(total_rows, dtype=(np.int32 if c == "category_id" else np.float32))
                for c in feature_cols
            }
            print(f"pos-teacher: {len(feature_cols)} donor features initialized", flush=True)

        n = df.height
        end = offset + n
        for col in feature_cols:
            arr = df[col].to_numpy()
            if col == "category_id":
                cols_buf[col][offset:end] = np.nan_to_num(arr, nan=-1).astype(np.int32, copy=False)
            else:
                f = arr.astype(np.float32, copy=False)
                if not np.isfinite(f).all():
                    f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
                cols_buf[col][offset:end] = f
        y_gain[offset:end] = df["_y_gain"].to_numpy()
        group_ids[offset:end] = df["feed_id"].to_numpy().astype(np.int32, copy=False)
        offset = end
        del df
        if idx % 10 == 0 or idx == len(donor_entries):
            print(f"pos-teacher donor load: {idx}/{len(donor_entries)} shards", flush=True)

    assert feature_cols is not None and offset == total_rows
    if len(np.unique(group_ids)) != 1 + int(np.count_nonzero(np.diff(group_ids))):
        raise ValueError("group_ids must form strictly contiguous feed_id blocks for CatBoostRanker")
    write_json(cols_path, feature_cols)
    df_x = pd.DataFrame(cols_buf, copy=False)
    cat_cols = ["category_id"] if "category_id" in feature_cols else []
    pool = Pool(df_x, label=y_gain, group_id=group_ids, cat_features=cat_cols)

    print("pos-teacher: training GPU CatBoostRanker model_qrmse (d8, lr=0.06, iters=1500)...", flush=True)
    model_qrmse = CatBoostRanker(
        loss_function="QueryRMSE",
        depth=8,
        learning_rate=0.06,
        iterations=1500,
        l2_leaf_reg=10.0,
        random_seed=42,
        task_type="GPU",
        border_count=254,
        one_hot_max_size=100,
        metric_period=200,
        verbose=200,
    )
    model_qrmse.fit(pool)
    model_qrmse.save_model(str(qrmse_path))
    qrmse_imp = sorted(
        zip(model_qrmse.get_feature_importance(pool), feature_cols), reverse=True
    )[:10]
    print(
        "pos-teacher QueryRMSE top10 importances:",
        [(c, round(float(v), 2)) for v, c in qrmse_imp],
        flush=True,
    )

    print("pos-teacher: training GPU CatBoostRanker model_yeti (d7, lr=0.06, iters=1200)...", flush=True)
    model_yeti = CatBoostRanker(
        loss_function="YetiRank",
        depth=7,
        learning_rate=0.06,
        iterations=1200,
        l2_leaf_reg=10.0,
        random_seed=43,
        task_type="GPU",
        border_count=254,
        one_hot_max_size=100,
        metric_period=200,
        verbose=200,
    )
    model_yeti.fit(pool)
    model_yeti.save_model(str(yeti_path))
    yeti_imp = sorted(
        zip(model_yeti.get_feature_importance(pool), feature_cols), reverse=True
    )[:10]
    print(
        "pos-teacher YetiRank top10 importances:",
        [(c, round(float(v), 2)) for v, c in yeti_imp],
        flush=True,
    )
    del pool, df_x, cols_buf, y_gain, group_ids
    gc.collect()

    return model_qrmse, model_yeti, feature_cols


def _export_run_outputs(
    prepared_dir: Path,
    prepared_manifest: dict,
    cache_dir: Path,
    cache_manifest: dict,
    run_dir: Path,
    sample_sub: Path,
    n_features: int,
) -> dict:
    """Build val_pred.parquet, test_pred.parquet, metrics.json for runs/pos-teacher-032."""
    val_parts = prepared_manifest["splits"]["val"]["parts"]
    val_cands = pl.concat(
        [
            pl.read_parquet(
                prepared_dir / p["candidate_path"],
                columns=["feed_id", "candidate_index", "item_id", "timestamp", "target"],
            )
            for p in val_parts
        ]
    )
    val_pt = pl.concat(
        [
            pl.read_parquet(
                cache_dir / cache_manifest["splits"]["val"]["parts"][p["source"]]["path"],
                columns=["feed_id", "candidate_index", "item_id", "pt_gain", "pt_pct", "pt_gain_z", "pt_pct_z"],
            )
            for p in val_parts
        ]
    )
    val_df = val_cands.join(val_pt, on=["feed_id", "candidate_index", "item_id"], how="left", validate="1:1")
    if val_df.select(["pt_gain", "pt_pct", "pt_gain_z", "pt_pct_z"]).null_count().to_numpy().any():
        raise ValueError("Null pos-teacher predictions found on val")

    val_df = val_df.with_columns(
        (0.5 * pl.col("pt_gain_z") + 0.5 * pl.col("pt_pct_z")).cast(pl.Float64).alias("score")
    ).sort(["feed_id", "candidate_index"])

    fids = val_df["feed_id"].to_numpy()
    ts = val_df["timestamp"].to_numpy()
    tgt = val_df["target"].to_numpy()
    pub_feeds, priv_feeds = pseudo_split(fids, ts)
    m_pub = mask_feeds(fids, pub_feeds)
    m_priv = mask_feeds(fids, priv_feeds)

    def _eval(score_arr: np.ndarray) -> dict[str, float]:
        out: dict[str, float] = {}
        for prefix, mask in (("all", None), ("pub", m_pub), ("priv", m_priv)):
            f_m = fids if mask is None else fids[mask]
            t_m = tgt if mask is None else tgt[mask]
            s_m = score_arr if mask is None else score_arr[mask]
            for k, v in all_variants(f_m, t_m, s_m).items():
                out[f"{prefix}_{k}"] = float(v)
        return out

    m_blend = _eval(val_df["score"].to_numpy())
    m_gain = _eval(val_df["pt_gain"].to_numpy())
    m_pct = _eval(val_df["pt_pct"].to_numpy())

    val_pred = val_df.select(["feed_id", "item_id", "target", "score"]).with_columns(
        [
            pl.col("feed_id").cast(pl.Int32),
            pl.col("item_id").cast(pl.Int32),
            pl.col("target").cast(pl.Int8),
            pl.col("score").cast(pl.Float64),
        ]
    )
    write_parquet_atomic(val_pred, run_dir / "val_pred.parquet")

    test_parts = prepared_manifest["splits"]["test"]["parts"]
    test_pt = pl.concat(
        [
            pl.read_parquet(
                cache_dir / cache_manifest["splits"]["test"]["parts"][p["source"]]["path"],
                columns=["feed_id", "item_id", "pt_gain_z", "pt_pct_z"],
            )
            for p in test_parts
        ]
    ).with_columns((0.5 * pl.col("pt_gain_z") + 0.5 * pl.col("pt_pct_z")).cast(pl.Float32).alias("score"))

    tpl = pl.read_parquet(sample_sub, columns=["feed_id", "item_id"])
    test_pred = tpl.join(
        test_pt.select(["feed_id", "item_id", "score"]),
        on=["feed_id", "item_id"],
        how="left",
        validate="1:1",
    )
    if test_pred["score"].null_count() > 0 or not np.isfinite(test_pred["score"].to_numpy()).all():
        raise ValueError("Invalid or missing pos-teacher scores when aligning to sample_submission")
    test_pred = test_pred.with_columns(
        [
            pl.col("feed_id").cast(pl.Int32),
            pl.col("item_id").cast(pl.Int32),
            pl.col("score").cast(pl.Float32),
        ]
    )
    write_parquet_atomic(test_pred, run_dir / "test_pred.parquet")

    metrics = {
        "n_features": n_features,
        "score_formula": "0.5 * pt_gain_z + 0.5 * pt_pct_z",
        **m_blend,
        "components": {"pt_gain": m_gain, "pt_pct": m_pct},
    }
    write_json(run_dir / "metrics.json", metrics)
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description="Train disjoint pos-teacher and materialize feature cache")
    p.add_argument("--prepared", default="data/prepared")
    p.add_argument("--features-dir", default="data/features")
    p.add_argument("--output", default="data/features/pos-teacher-032")
    p.add_argument("--run-dir", default="runs/pos-teacher-032")
    p.add_argument("--sample-submission", default="competition/sample_submission.parquet")
    p.add_argument("--train-parts", type=int, default=32)
    p.add_argument("--donor-block", default="train-blocks/block-032-064")
    p.add_argument("--donor-cosine", default="cosine-blocks-096")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    prepared_dir = Path(args.prepared).resolve()
    features_dir = Path(args.features_dir).resolve()
    output_dir = Path(args.output).resolve()
    run_dir = Path(args.run_dir).resolve()

    prepared_manifest = json.loads((prepared_dir / "manifest.json").read_text(encoding="utf-8"))
    excluded, donor_entries = _find_disjoint_donors(
        prepared_manifest, features_dir, args.donor_block, args.train_parts
    )
    model_gain, model_pct, feature_cols = _fit_or_load_models(
        prepared_dir,
        features_dir,
        run_dir,
        donor_entries,
        args.donor_block,
        args.donor_cosine,
        resume=args.resume,
    )

    infer_sources = [features_dir / "cosine", features_dir / "infeed", features_dir / "history"]

    def _build_part(split: str, entry: dict, _state: object) -> pl.DataFrame:
        df = _load_shard_features(prepared_dir, entry, split, infer_sources, include_pos=False)
        x_pdf = _to_pandas_features(df, feature_cols)
        p_gain = np.nan_to_num(model_gain.predict(x_pdf), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        p_pct = np.nan_to_num(model_pct.predict(x_pdf), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        keys = df.select(["split", "feed_id", "candidate_index", "item_id"])
        del df, x_pdf
        scored = keys.with_columns(
            [
                pl.Series("pt_gain", p_gain, dtype=pl.Float32),
                pl.Series("pt_pct", p_pct, dtype=pl.Float32),
            ]
        )
        g = pl.col("pt_gain")
        pct = pl.col("pt_pct")
        n_feed = pl.len().over("feed_id").cast(pl.Float32)
        g_std = g.std(ddof=0).over("feed_id").fill_nan(0.0).fill_null(0.0)
        p_std = pct.std(ddof=0).over("feed_id").fill_nan(0.0).fill_null(0.0)
        scored = scored.with_columns(
            [
                ((g - g.mean().over("feed_id")) / (g_std + 1e-6))
                .fill_nan(0.0)
                .fill_null(0.0)
                .cast(pl.Float32)
                .alias("pt_gain_z"),
                (g.rank(method="average").over("feed_id").cast(pl.Float32) / n_feed)
                .clip(0.0, 1.0)
                .fill_nan(0.0)
                .fill_null(0.0)
                .cast(pl.Float32)
                .alias("pt_gain_rank"),
                (g - g.max().over("feed_id"))
                .clip(upper_bound=0.0)
                .fill_nan(0.0)
                .fill_null(0.0)
                .cast(pl.Float32)
                .alias("pt_gain_dmax"),
                ((pct - pct.mean().over("feed_id")) / (p_std + 1e-6))
                .fill_nan(0.0)
                .fill_null(0.0)
                .cast(pl.Float32)
                .alias("pt_pct_z"),
            ]
        )
        return scored.select([*KEY_COLS, *PT_NAMES]).sort(["feed_id", "candidate_index"])

    extra = {
        "donor_block": args.donor_block,
        "donor_cosine": args.donor_cosine,
        "donor_sources": sorted(e["source"] for e in donor_entries),
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "model_gain": "CatBoostRanker(QueryRMSE, depth=8, lr=0.06, iterations=1500, l2=10, seed=42)",
        "model_pct": "CatBoostRanker(YetiRank, depth=7, lr=0.06, iterations=1200, l2=10, seed=43)",
    }
    cache_manifest = build_cache(
        prepared_dir,
        output_dir,
        train_parts=args.train_parts,
        group="pos_teacher",
        names=PT_NAMES,
        domains=PT_DOMAINS,
        producer=__file__,
        extra=extra,
        build_part=_build_part,
        resume=args.resume,
        excluded=excluded,
    )
    with cache_lock(output_dir):
        mpath = output_dir / "manifest.json"
        m = json.loads(mpath.read_bytes())
        m["target_policy"] = POLICY
        write_json(mpath, m)
        cache_manifest = m

    metrics = _export_run_outputs(
        prepared_dir,
        prepared_manifest,
        output_dir,
        cache_manifest,
        run_dir,
        Path(args.sample_submission).resolve(),
        len(feature_cols),
    )
    print(
        f"pos-teacher-032 VAL: all@30={metrics['all_ndcg@30']:.5f} "
        f"pub@30={metrics['pub_ndcg@30']:.5f} priv@30={metrics['priv_ndcg@30']:.5f} "
        f"(gain@30={metrics['components']['pt_gain']['all_ndcg@30']:.5f}, "
        f"pct@30={metrics['components']['pt_pct']['all_ndcg@30']:.5f})",
        flush=True,
    )


if __name__ == "__main__":
    main()
