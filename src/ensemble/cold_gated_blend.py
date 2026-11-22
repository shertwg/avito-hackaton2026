"""Честный 4-fold OOF Cold-Gated Blend: условное смешивание базового ансамбля и Cold-Start специалиста.

Мотивация:
- В базовых GBDT донорские товарные признаки (i2i/wi2i/is_item_*) перехватывают ~40% сплитов на train
  из-за высокой внутритрейновой корреляции (+0.29), но зануляются на холодных объявлениях (35.1% val, 46.6% test).
- Специалист без донорских item_id (cb32-cold-zeroi2i / lgb32-cold-zeroi2i) тратит 100% ёмкости деревьев
  на контент, эмбеддинги, внутриленточные статистики и категории (корреляция с базовым блендом ~0.919).
- На каждом из 4 временных фолдов val (src.common.cv.time_folds) по 3 тренировочным фолдам подбираются
  всего 2 параметра (a_cold, a_warm) на грубой регуляризованной сетке и усредняются для test.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from src.common.cv import mask_feeds, pseudo_split, time_folds
from src.common.data import load_labels
from src.common.metric import FastNDCGEvaluator
from src.common.scores import feed_zscore
from src.ensemble.meta_blend import paired_bootstrap
from src.gbdt.prepared import load_prepared

KEYS = ["feed_id", "item_id"]


def _load_cold_mask(
    labels_df: pl.DataFrame,
    prepared: str,
    split: str,
    itemstats_root: str,
    wi2i_root: str,
    train_parts: int = 32,
) -> np.ndarray:
    feat_df, _ = load_prepared(
        prepared,
        split,
        [itemstats_root, wi2i_root],
        train_parts=train_parts,
    )
    feat_df = feat_df.select("feed_id", "item_id", "is_item_impressions", "wi2i_all_max_50").cast(
        {"feed_id": pl.Int32, "item_id": pl.Int32}
    )
    joined = labels_df.select(KEYS).join(feat_df, on=KEYS, how="left")
    if joined["is_item_impressions"].null_count() or joined["wi2i_all_max_50"].null_count():
        raise ValueError(f"missing cold-gate features after join on split={split}")
    return (joined["is_item_impressions"].to_numpy() == 0) & (joined["wi2i_all_max_50"].to_numpy() == 0)


def _load_zscore_runs(labels_df: pl.DataFrame, runs: list[Path], which: str) -> np.ndarray:
    f = labels_df["feed_id"].to_numpy()
    zs = []
    for r in runs:
        p = pl.read_parquet(r / f"{which}_pred.parquet").select(KEYS + ["score"]).cast(
            {"feed_id": pl.Int32, "item_id": pl.Int32}
        )
        joined = labels_df.select(KEYS).join(p, on=KEYS, how="left")
        if joined["score"].null_count():
            raise ValueError(f"{r}/{which}_pred.parquet: nulls after join")
        zs.append(feed_zscore(f, joined["score"].to_numpy()).astype(np.float64))
    if len(zs) == 1:
        return zs[0]
    return feed_zscore(f, np.mean(zs, axis=0)).astype(np.float64)


def main(argv=None) -> None:
    t0 = time.perf_counter()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="competition")
    ap.add_argument("--prepared", default="data/prepared")
    ap.add_argument("--base-run", required=True, help="базовый OOF ансамбль (например runs/meta-blend-v43b-nn32-posf)")
    ap.add_argument("--cold-runs", nargs="+", required=True, help="один или несколько Cold-Start специалистов")
    ap.add_argument("--itemstats-root", default="data/features/itemstats-032")
    ap.add_argument("--wi2i-root", default="data/features/weighted-item2item-032")
    ap.add_argument("--train-parts", type=int, default=32)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--l2-reg", type=float, default=0.002)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    base_run = Path(a.base_run)
    cold_runs = [Path(r) for r in a.cold_runs]

    val = load_labels(a.data, "val").cast({"feed_id": pl.Int32, "item_id": pl.Int32}).sort("feed_id", maintain_order=True)
    f = val["feed_id"].to_numpy()
    t = val["target"].to_numpy()
    ts = val["timestamp"].to_numpy()
    pub, priv = pseudo_split(f, ts)
    m_pub, m_priv = mask_feeds(f, pub), mask_feeds(f, priv)

    is_cold_val = _load_cold_mask(val, a.prepared, "val", a.itemstats_root, a.wi2i_root, a.train_parts)
    s_base = _load_zscore_runs(val, [base_run], "val")
    s_cold = _load_zscore_runs(val, cold_runs, "val")

    ev_all = FastNDCGEvaluator(f, t, cutoff=30)
    ev_pub = FastNDCGEvaluator(f[m_pub], t[m_pub], cutoff=30)
    ev_priv = FastNDCGEvaluator(f[m_priv], t[m_priv], cutoff=30)

    uf, fold = time_folds(f, ts, a.folds)
    fold_of = dict(zip(uf.tolist(), fold.tolist()))
    row_fold = np.array([fold_of[i] for i in f])

    grid_cold = (0.0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.22)
    grid_warm = (0.0, 0.02, 0.04, 0.06)

    oof = np.zeros(len(f), dtype=np.float64)
    fold_params: list[dict[str, float]] = []
    for k in range(a.folds):
        held = row_fold == k
        tr = ~held
        ev_tr = FastNDCGEvaluator(f[tr], t[tr], cutoff=30)
        bc = ev_tr.compact_matrix(s_base[tr, None])[:, 0]
        cc = ev_tr.compact_matrix(s_cold[tr, None])[:, 0]
        mc = ev_tr.compact_matrix(is_cold_val[tr, None].astype(np.float32))[:, 0] > 0.5

        best_cfg = (0.0, 0.0)
        best_obj = -1e9
        for ac in grid_cold:
            s_c = (1.0 - ac) * bc + ac * cc
            for aw in grid_warm:
                s_w = (1.0 - aw) * bc + aw * cc
                s_try = np.where(mc, s_c, s_w)
                pen = a.l2_reg * (ac * ac + aw * aw)
                obj = ev_tr.mean_from_compact(s_try) - pen
                if obj > best_obj + 1e-8:
                    best_obj = obj
                    best_cfg = (ac, aw)

        ac, aw = best_cfg
        fold_params.append({"a_cold": float(ac), "a_warm": float(aw)})
        oof[held] = np.where(
            is_cold_val[held],
            (1.0 - ac) * s_base[held] + ac * s_cold[held],
            (1.0 - aw) * s_base[held] + aw * s_cold[held],
        )
        ev_h = FastNDCGEvaluator(f[held], t[held], cutoff=30)
        print(
            f"[cold-gated] fold {k+1}/{a.folds} a_cold={ac:.2f} a_warm={aw:.2f} "
            f"held_ndcg@30={ev_h.mean(oof[held]):.5f} (base={ev_h.mean(s_base[held]):.5f})",
            flush=True,
        )

    ac_avg = float(np.mean([p["a_cold"] for p in fold_params]))
    aw_avg = float(np.mean([p["a_warm"] for p in fold_params]))

    te = load_labels(a.data, "test").cast({"feed_id": pl.Int32, "item_id": pl.Int32}).sort("feed_id", maintain_order=True)
    is_cold_te = _load_cold_mask(te, a.prepared, "test", a.itemstats_root, a.wi2i_root, a.train_parts)
    st_base = _load_zscore_runs(te, [base_run], "test")
    st_cold = _load_zscore_runs(te, cold_runs, "test")
    st_out = np.where(
        is_cold_te,
        (1.0 - ac_avg) * st_base + ac_avg * st_cold,
        (1.0 - aw_avg) * st_base + aw_avg * st_cold,
    ).astype(np.float32)

    pl.DataFrame({"feed_id": te["feed_id"], "item_id": te["item_id"], "score": st_out}).write_parquet(
        out / "test_pred.parquet"
    )
    val.select("feed_id", "item_id", "target").with_columns(pl.Series("score", oof.astype(np.float32))).write_parquet(
        out / "val_pred.parquet"
    )

    report = {
        "base_run": str(base_run),
        "cold_runs": [str(r) for r in cold_runs],
        "val_cold_share": float(is_cold_val.mean()),
        "test_cold_share": float(is_cold_te.mean()),
        "corr_overall": float(np.corrcoef(s_base, s_cold)[0, 1]),
        "corr_cold": float(np.corrcoef(s_base[is_cold_val], s_cold[is_cold_val])[0, 1]),
        "corr_warm": float(np.corrcoef(s_base[~is_cold_val], s_cold[~is_cold_val])[0, 1]),
        "base_metrics": {
            "pub": float(ev_pub.mean(s_base[m_pub])),
            "priv": float(ev_priv.mean(s_base[m_priv])),
            "all": float(ev_all.mean(s_base)),
        },
        "cold_metrics": {
            "pub": float(ev_pub.mean(s_cold[m_pub])),
            "priv": float(ev_priv.mean(s_cold[m_priv])),
            "all": float(ev_all.mean(s_cold)),
        },
        "oof_cold_gated": {
            "pub": float(ev_pub.mean(oof[m_pub])),
            "priv": float(ev_priv.mean(oof[m_priv])),
            "all": float(ev_all.mean(oof)),
        },
        "params_avg": {"a_cold": round(ac_avg, 4), "a_warm": round(aw_avg, 4)},
        "params_per_fold": fold_params,
        "bootstrap_vs_base_all": paired_bootstrap(f, t, s_base, oof),
        "bootstrap_vs_base_priv": paired_bootstrap(f[m_priv], t[m_priv], s_base[m_priv], oof[m_priv]),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"[base]       all={report['base_metrics']['all']:.6f}  pub={report['base_metrics']['pub']:.6f}  priv={report['base_metrics']['priv']:.6f}"
    )
    print(
        f"[cold-gated] all={report['oof_cold_gated']['all']:.6f}  pub={report['oof_cold_gated']['pub']:.6f}  priv={report['oof_cold_gated']['priv']:.6f}"
    )
    print("params_avg:", report["params_avg"], f"(val_cold={report['val_cold_share']:.3f}, test_cold={report['test_cold_share']:.3f})")
    print("bootstrap vs base (all):", report["bootstrap_vs_base_all"])
    print("bootstrap vs base (priv):", report["bootstrap_vs_base_priv"])


if __name__ == "__main__":
    main()
