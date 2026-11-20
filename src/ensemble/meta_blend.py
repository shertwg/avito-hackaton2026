"""Честный 4-fold OOF оптимизатор ансамбля: гибрид Z-score + RankGauss с L2-регуляризацией.

В отличие от blend.py (грубая сетка только на pseudo-public), meta_blend.py:
1. Подбирает веса строго out-of-fold на 4 временных блоках val (src.common.cv.time_folds):
   на каждом фолде k веса оптимизируются ТОЛЬКО на 3 других фолдах (3/4 val), а оцениваются
   на отложенном фолде k. Так и pub@30, и priv@30, и all@30 являются 100% честным OOF!
2. Использует гибридное представление внутри ленты: (1 - alpha) * z_score + alpha * rank_gauss,
   что сглаживает выбросы и стабилизирует NDCG@30.
3. Регуляризует веса (L2 к равномерным весам) и усредняет веса по 4 фолдам для test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.special import ndtri

from src.common.cv import mask_feeds, pseudo_split, time_folds
from src.common.data import load_labels
from src.common.metric import mean_ndcg, per_feed_ndcg
from src.common.scores import feed_rank01, feed_zscore

KEYS = ["feed_id", "item_id"]


def collect(base: pl.DataFrame, runs: list[Path], which: str) -> tuple[pl.DataFrame, list[str]]:
    names = []
    for r in runs:
        name = r.name
        names.append(name)
        p = pl.read_parquet(r / f"{which}_pred.parquet").select(KEYS + ["score"]).rename({"score": name})
        p = p.cast({"feed_id": pl.Int32, "item_id": pl.Int32})
        base = base.join(p, on=KEYS, how="left")
        if base[name].null_count():
            raise ValueError(f"{r}/{which}_pred.parquet: пропуски после join")
    return base, names


def hybrid_matrix(df: pl.DataFrame, names: list[str], rank_alpha: float = 0.25) -> np.ndarray:
    """Внутриленточная смесь z-score и rank-gauss (probit of percentile rank)."""
    f = df["feed_id"].to_numpy()
    cols = []
    for n in names:
        raw = df[n].to_numpy()
        z = feed_zscore(f, raw)
        if rank_alpha > 0.0:
            r01 = feed_rank01(f, raw)
            # Клиппируем (0, 1) -> нормальные квантили с дисперсией ~1
            rg = ndtri(np.clip(r01, 1e-4, 1.0 - 1e-4)).astype(np.float32)
            rg = feed_zscore(f, rg)
            cols.append(((1.0 - rank_alpha) * z + rank_alpha * rg).astype(np.float32))
        else:
            cols.append(z.astype(np.float32))
    return np.stack(cols, axis=1)


def fit_weights_regularized(
    feed_id: np.ndarray,
    target: np.ndarray,
    X: np.ndarray,
    l2_reg: float = 0.002,
    n_restarts: int = 12,
    seed: int = 42,
) -> np.ndarray:
    """Оптимизирует неотрицательные веса на симплексе для NDCG@30 - l2_reg * ||w - 1/M||^2."""
    m = X.shape[1]
    w_unif = np.ones(m, dtype=np.float64) / m
    rng = np.random.default_rng(seed)

    def eval_w(w: np.ndarray) -> float:
        w = np.maximum(w, 0.0)
        s = w.sum()
        if s <= 0.0:
            return -1e9
        w = w / s
        score = X @ w.astype(np.float32)
        val = mean_ndcg(feed_id, target, score, cutoff=30)
        pen = l2_reg * float(np.sum((w - w_unif) ** 2))
        return val - pen

    # 1. Быстрый стохастический разогрев (Dirichlet + координатные шаги)
    best_w = w_unif.copy()
    best_obj = eval_w(best_w)

    # Поодиночный и парный старт + Дирихле
    cands = [w_unif]
    for _ in range(n_restarts * 4):
        cands.append(rng.dirichlet(np.ones(m) * 1.5))
    for w in cands:
        obj = eval_w(w)
        if obj > best_obj:
            best_obj, best_w = obj, w.copy()

    # 2. Многоступенчатый покоординатный поиск с адаптивным шагом (непрерывный!)
    w = best_w.copy()
    for step in (0.16, 0.08, 0.04, 0.02, 0.01):
        for _ in range(3):
            improved = False
            for i in range(m):
                for delta in (-step, step):
                    cand = w.copy()
                    cand[i] = max(0.0, cand[i] + delta)
                    if cand.sum() <= 0:
                        continue
                    cand /= cand.sum()
                    obj = eval_w(cand)
                    if obj > best_obj + 1e-7:
                        best_obj, w, improved = obj, cand, True
            if not improved:
                break
    return w / w.sum()


def paired_bootstrap(feed_id: np.ndarray, target: np.ndarray, s_a: np.ndarray, s_b: np.ndarray,
                     reps: int = 1000, seed: int = 42) -> dict:
    _, a = per_feed_ndcg(feed_id, target, s_a, cutoff=30)
    _, b = per_feed_ndcg(feed_id, target, s_b, cutoff=30)
    delta = b - a
    rng = np.random.default_rng(seed)
    draws = np.array([rng.choice(delta, size=len(delta), replace=True).mean() for _ in range(reps)])
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {
        "delta_mean": float(delta.mean()),
        "ci95": [float(lo), float(hi)],
        "p_positive": float((draws > 0).mean()),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="competition")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--rank-alpha", type=float, default=0.25, help="доля RankGauss в гибриде с Z-score (0..1)")
    ap.add_argument("--l2-reg", type=float, default=0.002, help="L2-штраф к равномерным весам")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = [Path(r) for r in a.runs]

    lab = load_labels(a.data, "val").cast({"feed_id": pl.Int32, "item_id": pl.Int32})
    val, names = collect(lab, runs, "val")
    val = val.sort("feed_id", maintain_order=True)
    f = val["feed_id"].to_numpy()
    t = val["target"].to_numpy()
    ts = val["timestamp"].to_numpy()
    pub, priv = pseudo_split(f, ts)
    m_pub, m_priv = mask_feeds(f, pub), mask_feeds(f, priv)

    X = hybrid_matrix(val, names, rank_alpha=a.rank_alpha)
    w_mean = np.ones(len(names), dtype=np.float64) / len(names)
    s_mean = X @ w_mean

    uf, fold = time_folds(f, ts, a.folds)
    fold_of = dict(zip(uf.tolist(), fold.tolist()))
    row_fold = np.array([fold_of[i] for i in f])

    oof = np.zeros(len(f), dtype=np.float64)
    fold_weights = []
    for k in range(a.folds):
        held = row_fold == k
        tr = ~held
        wk = fit_weights_regularized(f[tr], t[tr], X[tr], l2_reg=a.l2_reg, seed=a.seed + k)
        fold_weights.append(wk)
        oof[held] = X[held] @ wk
        print(f"[oof-blend] fold {k+1}/{a.folds} held_ndcg@30={mean_ndcg(f[held], t[held], oof[held], 30):.5f}", flush=True)

    w_avg = np.mean(fold_weights, axis=0)
    w_avg /= w_avg.sum()

    res = {
        "rank_alpha": a.rank_alpha,
        "l2_reg": a.l2_reg,
        "models": {
            n: {
                "pub": float(mean_ndcg(f[m_pub], t[m_pub], X[m_pub, i], 30)),
                "priv": float(mean_ndcg(f[m_priv], t[m_priv], X[m_priv, i], 30)),
                "all": float(mean_ndcg(f, t, X[:, i], 30)),
            }
            for i, n in enumerate(names)
        },
        "mean_hybrid": {
            "pub": float(mean_ndcg(f[m_pub], t[m_pub], s_mean[m_pub], 30)),
            "priv": float(mean_ndcg(f[m_priv], t[m_priv], s_mean[m_priv], 30)),
            "all": float(mean_ndcg(f, t, s_mean, 30)),
        },
        "oof_weighted": {
            "pub": float(mean_ndcg(f[m_pub], t[m_pub], oof[m_pub], 30)),
            "priv": float(mean_ndcg(f[m_priv], t[m_priv], oof[m_priv], 30)),
            "all": float(mean_ndcg(f, t, oof, 30)),
        },
        "bootstrap_vs_mean_all": paired_bootstrap(f, t, s_mean, oof),
        "bootstrap_vs_mean_priv": paired_bootstrap(f[m_priv], t[m_priv], s_mean[m_priv], oof[m_priv]),
        "weights_avg": dict(zip(names, np.round(w_avg, 4).tolist())),
        "weights_per_fold": [dict(zip(names, np.round(wk, 4).tolist())) for wk in fold_weights],
    }

    # Test prediction using averaged fold weights
    te_keys = load_labels(a.data, "test").cast({"feed_id": pl.Int32, "item_id": pl.Int32})
    te, _ = collect(te_keys, runs, "test")
    te = te.sort("feed_id", maintain_order=True)
    Xt = hybrid_matrix(te, names, rank_alpha=a.rank_alpha)
    st = (Xt @ w_avg).astype(np.float32)

    pl.DataFrame({"feed_id": te["feed_id"], "item_id": te["item_id"], "score": st}).write_parquet(out / "test_pred.parquet")
    val.select("feed_id", "item_id", "target").with_columns(pl.Series("score", oof.astype(np.float32))).write_parquet(out / "val_pred.parquet")
    (out / "report.json").write_text(json.dumps(res, indent=2))

    print(f"[mean_hybrid]  all={res['mean_hybrid']['all']:.5f}  pub={res['mean_hybrid']['pub']:.5f}  priv={res['mean_hybrid']['priv']:.5f}")
    print(f"[oof_weighted] all={res['oof_weighted']['all']:.5f}  pub={res['oof_weighted']['pub']:.5f}  priv={res['oof_weighted']['priv']:.5f}")
    print("bootstrap vs mean (all):", res["bootstrap_vs_mean_all"])
    print("weights_avg:", res["weights_avg"])


if __name__ == "__main__":
    main()
