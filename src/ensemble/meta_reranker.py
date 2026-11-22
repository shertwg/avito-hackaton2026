"""2nd-Level Listwise Meta-Reranker (Wave 4): стекинг разнородных моделей + контекст ленты/кандидата.

Строго 4-fold OOF на всех 14 423 лентах val (src.common.cv.time_folds):
1. Для каждой модели 1-го уровня строит внутриленточные проекции:
   - z_score, rank_gauss (probit), gap_to_top1 (z - max_feed(z))
2. Считает кросс-модельные статистики по кандидату:
   - mean_z, std_z (мера несогласия моделей), min_z, max_z, max_minus_mean
3. Добавляет контекст ленты и сильные опорные признаки (из val/test паркетов и кешей, если переданы):
   - feed_size, log1p(feed_size), n_unique_cats, incat_count, incat_share
4. Обучает Shallow LightGBM LambdaRank@30 (с сильной регуляризацией) и опционально CatBoost QSM,
   смешивая их OOF с лучшим линейным гибридом Meta-A (alpha * meta_gbdt + (1 - alpha) * linear_hybrid).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.special import ndtri

from src.common.cv import mask_feeds, pseudo_split, time_folds
from src.common.data import load_labels
from src.common.metric import mean_ndcg
from src.common.scores import feed_rank01, feed_zscore
from src.ensemble.meta_blend import collect, paired_bootstrap

KEYS = ["feed_id", "item_id"]


def build_meta_features(df: pl.DataFrame, names: list[str], extra_cols: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Строит матрицу признаков 2-го уровня и возвращает (X_meta, mean_hybrid_score)."""
    f = df["feed_id"].to_numpy()
    zs, rgs, gaps = [], [], []
    for n in names:
        raw = df[n].to_numpy()
        z = feed_zscore(f, raw).astype(np.float32)
        r01 = feed_rank01(f, raw)
        rg = feed_zscore(f, ndtri(np.clip(r01, 1e-4, 1.0 - 1e-4)).astype(np.float32)).astype(np.float32)
        zs.append(z)
        rgs.append(rg)

    Z = np.stack(zs, axis=1)
    RG = np.stack(rgs, axis=1)
    # Gap to top-1 in each feed per model
    tmp_df = pl.DataFrame({"feed_id": f, **{f"_z{i}": Z[:, i] for i in range(len(names))}})
    max_exprs = [(pl.col(f"_z{i}") - pl.col(f"_z{i}").max().over("feed_id")).alias(f"_g{i}") for i in range(len(names))]
    G = tmp_df.with_columns(max_exprs).select([f"_g{i}" for i in range(len(names))]).to_numpy().astype(np.float32)

    z_mean = Z.mean(axis=1, keepdims=True)
    z_std = Z.std(axis=1, keepdims=True)
    z_min = Z.min(axis=1, keepdims=True)
    z_max = Z.max(axis=1, keepdims=True)
    rg_mean = RG.mean(axis=1, keepdims=True)
    rg_std = RG.std(axis=1, keepdims=True)

    # Context features from feed structure
    feed_stats = (
        df.select("feed_id")
        .with_columns(
            pl.len().over("feed_id").cast(pl.Float32).alias("_fsize"),
        )
        .with_columns(pl.col("_fsize").log1p().alias("_log_fsize"))
    )
    ctx_cols = ["_fsize", "_log_fsize"]
    if "category_id" in df.columns:
        feed_stats = df.select("feed_id", "category_id").with_columns(
            pl.len().over("feed_id").cast(pl.Float32).alias("_fsize"),
            pl.len().over(["feed_id", "category_id"]).cast(pl.Float32).alias("_incat_n"),
            pl.col("category_id").n_unique().over("feed_id").cast(pl.Float32).alias("_n_cats"),
        ).with_columns(
            pl.col("_fsize").log1p().alias("_log_fsize"),
            (pl.col("_incat_n") / pl.col("_fsize")).alias("_incat_share"),
        )
        ctx_cols = ["_fsize", "_log_fsize", "_incat_n", "_incat_share", "_n_cats"]

    parts = [Z, RG, G, z_mean, z_std, z_min, z_max, z_max - z_mean, rg_mean, rg_std,
             feed_stats.select(ctx_cols).to_numpy().astype(np.float32)]
    if extra_cols:
        parts.append(df.select([pl.col(c).cast(pl.Float32) for c in extra_cols]).to_numpy().astype(np.float32))

    X_meta = np.column_stack(parts).astype(np.float32)
    hybrid = (0.75 * Z + 0.25 * RG).mean(axis=1).astype(np.float32)
    return X_meta, hybrid


def attach_context_cols(base: pl.DataFrame, data_dir: str, split: str) -> pl.DataFrame:
    """Подтягивает category_id и f1..f5 из сырых паркетов (быстро, без признаков)."""
    from src.common.data import list_parts
    chunks = []
    for p in list_parts(data_dir, split):
        chunks.append(
            pl.read_parquet(p, columns=["feed_id", "item_id", "category_id", "f1", "f2", "f3", "f4", "f5"])
            .explode(["item_id", "category_id", "f1", "f2", "f3", "f4", "f5"])
            .cast({"feed_id": pl.Int32, "item_id": pl.Int32, "category_id": pl.Int32})
        )
    ctx = pl.concat(chunks)
    return base.join(ctx, on=KEYS, how="left")


def fit_lgb_meta(X_tr: np.ndarray, y_tr: np.ndarray, f_tr: np.ndarray, seed: int = 42,
                 lr: float = 0.03, rounds: int = 150, num_leaves: int = 7, min_data: int = 500):
    import lightgbm as lgb
    _, counts = np.unique(f_tr, return_counts=True)
    params = dict(
        objective="lambdarank",
        learning_rate=lr,
        num_leaves=num_leaves,
        min_data_in_leaf=min_data,
        feature_fraction=0.55,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=50.0,
        max_bin=63,
        seed=seed,
        verbose=-1,
        metric="ndcg",
        eval_at=[30],
    )
    return lgb.train(params, lgb.Dataset(X_tr, label=y_tr, group=counts), num_boost_round=rounds)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="competition")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--base-run", default=None, help="готовый 4-fold OOF бленд (Meta-A) в качестве базы для смешивания")
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--num-leaves", type=int, default=7)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = [Path(r) for r in a.runs]

    lab = load_labels(a.data, "val").cast({"feed_id": pl.Int32, "item_id": pl.Int32})
    val, names = collect(lab, runs, "val")
    val = attach_context_cols(val, a.data, "val").sort("feed_id", maintain_order=True)
    f, t, ts = val["feed_id"].to_numpy(), val["target"].to_numpy(), val["timestamp"].to_numpy()
    pub, priv = pseudo_split(f, ts)
    m_pub, m_priv = mask_feeds(f, pub), mask_feeds(f, priv)

    extra_cols = ["f1", "f2", "f3", "f4", "f5"]
    X_meta, s_base = build_meta_features(val, names, extra_cols=extra_cols)
    if a.base_run:
        bp = pl.read_parquet(Path(a.base_run) / "val_pred.parquet").select(KEYS + ["score"]).cast(
            {"feed_id": pl.Int32, "item_id": pl.Int32}
        )
        s_base = feed_zscore(f, val.select(KEYS).join(bp, on=KEYS, how="left")["score"].to_numpy())

    uf, fold = time_folds(f, ts, a.folds)
    fold_of = dict(zip(uf.tolist(), fold.tolist()))
    row_fold = np.array([fold_of[i] for i in f])

    oof_lgb = np.zeros(len(f), dtype=np.float64)
    fold_models = []
    for k in range(a.folds):
        held = row_fold == k
        tr = ~held
        m = fit_lgb_meta(X_meta[tr], t[tr], f[tr], seed=a.seed + k, lr=a.lr, rounds=a.rounds, num_leaves=a.num_leaves)
        fold_models.append(m)
        oof_lgb[held] = feed_zscore(f[held], m.predict(X_meta[held]))
        print(f"[meta-reranker] fold {k+1}/{a.folds} lgb_ndcg@30={mean_ndcg(f[held], t[held], oof_lgb[held], 30):.5f}", flush=True)

    # Check blend alpha between linear hybrid/base and non-linear meta_lgb strictly on pub/all
    alphas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    grid = []
    for al in alphas:
        s = (1.0 - al) * s_base + al * oof_lgb
        grid.append((al, float(mean_ndcg(f[m_pub], t[m_pub], s[m_pub], 30)),
                     float(mean_ndcg(f[m_priv], t[m_priv], s[m_priv], 30)),
                     float(mean_ndcg(f, t, s, 30))))
        print(f"  alpha={al:.2f}: all={grid[-1][3]:.5f}  pub={grid[-1][1]:.5f}  priv={grid[-1][2]:.5f}", flush=True)

    # Select alpha by pub@30 (zero private tuning)
    best_al, best_pub, best_priv, best_all = max(grid, key=lambda x: x[1])
    s_best = ((1.0 - best_al) * s_base + best_al * oof_lgb).astype(np.float32)

    # Predict test using 4-fold ensemble (exact calibration match to OOF!)
    te_keys = load_labels(a.data, "test").cast({"feed_id": pl.Int32, "item_id": pl.Int32})
    te, _ = collect(te_keys, runs, "test")
    te = attach_context_cols(te, a.data, "test").sort("feed_id", maintain_order=True)
    Xt_meta, st_base = build_meta_features(te, names, extra_cols=extra_cols)
    ft = te["feed_id"].to_numpy()
    if a.base_run:
        btp = pl.read_parquet(Path(a.base_run) / "test_pred.parquet").select(KEYS + ["score"]).cast(
            {"feed_id": pl.Int32, "item_id": pl.Int32}
        )
        st_base = feed_zscore(ft, te.select(KEYS).join(btp, on=KEYS, how="left")["score"].to_numpy())
    st_lgb = feed_zscore(
        ft,
        np.mean([feed_zscore(ft, m_k.predict(Xt_meta)) for m_k in fold_models], axis=0),
    )
    st_best = ((1.0 - best_al) * st_base + best_al * st_lgb).astype(np.float32)

    pl.DataFrame({"feed_id": ft, "item_id": te["item_id"], "score": st_best}).write_parquet(out / "test_pred.parquet")
    val.select("feed_id", "item_id", "target").with_columns(pl.Series("score", s_best)).write_parquet(out / "val_pred.parquet")
    report = {
        "n_models": len(names),
        "n_meta_features": int(X_meta.shape[1]),
        "best_alpha_by_pub": best_al,
        "grid": [{"alpha": g[0], "pub": g[1], "priv": g[2], "all": g[3]} for g in grid],
        "bootstrap_vs_hybrid_all": paired_bootstrap(f, t, s_base, s_best),
        "bootstrap_vs_hybrid_priv": paired_bootstrap(f[m_priv], t[m_priv], s_base[m_priv], s_best[m_priv]),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"chosen alpha={best_al:.2f} -> all={best_all:.5f} pub={best_pub:.5f} priv={best_priv:.5f}")


if __name__ == "__main__":
    main()
