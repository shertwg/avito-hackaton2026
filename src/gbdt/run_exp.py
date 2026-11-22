"""Раннер экспериментов GBDT.

Режимы:
  holdout — обучение на train, early stopping на pseudo-public части val, метрики на val и test-предсказания
  cv      — K временных блоков val: каждый блок предсказывается моделью, обученной на train + остальной val
            (даёт OOF-предсказания на всём val для ансамбля/стекинга; итерации фиксированы)
  final   — обучение на train + весь val, предсказание test (метрик нет: val в обучении)

Работа с позицией (--pos-mode):
  none     — pos не используется
  feature  — pos как признак; на val/test подставляется --pos-const
  ipw      — веса позитивов 1/p(pos) (для listwise лоссов вроде QuerySoftMax)
  ipw_all  — веса 1/p(pos) на позитивы И негативы с нормировкой среднего к 1.0 (SNIPS для поточечных лоссов)
  topn     — обучаться только на строках train с pos < --topn (меньше смещение, меньше данных)
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import time
import warnings
from pathlib import Path

import numpy as np
import polars as pl

from src.common.cv import mask_feeds, pseudo_split, time_folds
from src.common.data import MODEL_FEATURES, load_split
from src.common.metric import all_variants
from src.common.scores import feed_zscore
from src.gbdt.extra_features import attach
from src.gbdt.models import Ranker, prepare_predict_input
from src.gbdt.weights import ipw_weights, position_ctr

_POINTWISE_LOSSES = {"binary", "cross_entropy", "Logloss", "CrossEntropy", "RMSE"}


def get_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="сырые данные competition/ (или подготовленные при --prepared)")
    ap.add_argument("--prepared", default=None, help="data/prepared Димы: читать готовые таблицы вместо сырых лент")
    ap.add_argument(
        "--feature-root",
        action="append",
        default=[],
        help="группа признаков Димы (data/features/cosine, data/features/infeed); можно повторять",
    )
    ap.add_argument("--train-parts", type=int, default=1, help="сколько train-шардов брать при --prepared")
    ap.add_argument("--train-start", type=int, help="начальная позиция непрерывного блока train-шардов при --prepared")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["holdout", "cv", "final"], default="holdout")
    ap.add_argument("--kind", choices=["catboost", "lightgbm"], default="catboost")
    ap.add_argument("--loss", default=None, help="YetiRank / YetiRankPairwise / QueryRMSE / lambdarank / binary ...")
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--iterations-from", default=None, help="metrics.json holdout-запуска: взять best_iteration")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task-type", default="CPU", choices=["CPU", "GPU"])
    ap.add_argument("--pos-mode", choices=["none", "feature", "ipw", "ipw_all", "topn"], default="none")
    ap.add_argument("--pos-const", type=float, default=0.0)
    ap.add_argument("--topn", type=int, default=50)
    ap.add_argument("--ipw-max", type=float, default=10.0)
    ap.add_argument("--val-weight", type=float, default=1.0, help="вес строк val при обучении на train+val")
    ap.add_argument(
        "--time-decay",
        type=float,
        default=0.0,
        help="экспоненциальное затухание веса train по дням: exp(-decay * age_days)",
    )
    ap.add_argument(
        "--drop-zero-grad-feeds",
        action="store_true",
        help="отбросить на train ленты без кликов (или со всеми кликами): нулевой градиент для ранкеров",
    )
    ap.add_argument("--folds", type=int, default=4, help="число временных блоков для режима cv")
    ap.add_argument("--model-params", default=None, help="JSON-файл с доп. параметрами модели (пишет tune.py)")
    ap.add_argument("--features-dir", default=None, help="папка с внешними признаками (extra_features.py)")
    ap.add_argument("--extra", nargs="*", default=[], help="группы внешних признаков, напр. anton avgust")
    ap.add_argument("--drop-cols-file", default=None, help="файл со списком колонок для удаления")
    ap.add_argument("--feed-interactions", action="store_true", help="добавить внутриленточные z-score/dmax и взаимодействия")
    ap.add_argument("--max-parts", type=int, default=None)
    ap.add_argument("--n-jobs", type=int, default=1, help="процессов при построении признаков")
    ap.add_argument("--cache", default=None, help="папка кэша таблиц признаков")
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--note", default="")
    ap.add_argument("--log", action="store_true", help="дописать строку в experiments/log.md (только для реальных данных)")
    args = ap.parse_args(argv)
    if args.train_start is not None and not args.prepared:
        ap.error("--train-start requires --prepared")
    if args.drop_zero_grad_feeds and (args.loss in _POINTWISE_LOSSES):
        ap.error(
            f"--drop-zero-grad-feeds нельзя использовать с поточечным лоссом {args.loss}: "
            "ленты без кликов несут ненулевой отрицательный градиент."
        )
    if args.pos_mode == "ipw" and (args.loss in _POINTWISE_LOSSES):
        warnings.warn(
            f"--pos-mode ipw взвешивает только позитивы и смещает баланс классов для поточечного лосса {args.loss}; "
            "предпочтительнее --pos-mode ipw_all или --pos-mode none.",
            stacklevel=2,
        )
    return args


def _filter_zero_grad(df: pl.DataFrame) -> pl.DataFrame:
    """Оставляет только ленты с хотя бы одним кликом и одним не-кликом (ненулевой градиент для listwise/pairwise)."""
    s = pl.col("target").sum().over("feed_id")
    n = pl.len().over("feed_id")
    return df.filter((s > 0) & (s < n))


_PROTECTED_COLS = {"feed_id", "item_id", "category_id", "target", "pos", "timestamp", "candidate_index", "split"}


def _load_drop_cols(path: str | None) -> set[str]:
    if not path:
        return set()
    cols = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            cols.add(s)
    return cols


def load_all(a: argparse.Namespace, need_test: bool) -> tuple[dict[str, pl.DataFrame], list[str]]:
    t_start = time.perf_counter()
    frames, extra_cols = {}, []
    splits = ["train", "val"] + (["test"] if need_test else [])
    base_feats = MODEL_FEATURES
    drop_zg = getattr(a, "drop_zero_grad_feeds", False)
    drop_cols = _load_drop_cols(getattr(a, "drop_cols_file", None)) - _PROTECTED_COLS
    for s in splits:
        if a.prepared:
            from src.gbdt.prepared import load_prepared

            df, base_feats = load_prepared(
                a.prepared, s, a.feature_root, a.train_parts, train_start=getattr(a, "train_start", None)
            )
        else:
            df = load_split(a.data, s, a.max_parts, a.n_jobs, a.cache)
        if s == "train" and drop_zg and "target" in df.columns:
            before = df.height
            df = _filter_zero_grad(df)
            print(
                f"[drop-zero-grad] train rows: {before:,} -> {df.height:,} ({100.0 * df.height / max(before, 1):.1f}%)",
                flush=True,
            )
        df, extra_cols = attach(df, a.features_dir, a.extra, s)
        if drop_cols:
            present = [c for c in df.columns if c in drop_cols]
            if present:
                df = df.drop(present)
                if s == "train":
                    print(f"[drop-cols] dropped {len(present)} raw cols", flush=True)
            base_feats = [c for c in base_feats if c not in drop_cols]
            extra_cols = [c for c in extra_cols if c not in drop_cols]
        fi_cols: list[str] = []
        if getattr(a, "feed_interactions", False):
            from src.gbdt.feed_interactions import add_feed_interactions

            df, fi_cols = add_feed_interactions(df)
            if drop_cols:
                present_fi = [c for c in fi_cols if c in drop_cols]
                if present_fi:
                    df = df.drop(present_fi)
                    fi_cols = [c for c in fi_cols if c not in drop_cols]
                    if s == "train":
                        print(f"[drop-cols] dropped {len(present_fi)} fi cols", flush=True)
        if "pos" not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.Int16).alias("pos"))
        df = df.with_columns(pl.col("category_id").cast(pl.Int32), pl.col("pos").cast(pl.Int16))
        if not df["feed_id"].is_sorted():
            df = df.sort("feed_id", maintain_order=True)
        frames[s] = df
    feats = base_feats + extra_cols + fi_cols
    print(f"[load] {len(feats)} features loaded across {splits} in {time.perf_counter() - t_start:.1f}s", flush=True)
    return frames, feats


def apply_pos(df: pl.DataFrame, a: argparse.Namespace, is_train: bool) -> pl.DataFrame:
    """Для режима feature: pos в train — настоящий (null -> -1), в val/test — константа."""
    if a.pos_mode != "feature":
        return df
    if is_train:
        return df.with_columns(pl.col("pos").fill_null(-1).cast(pl.Float32))
    return df.with_columns(pl.lit(a.pos_const).cast(pl.Float32).alias("pos"))


def _prepare_base_train(frames: dict[str, pl.DataFrame], a: argparse.Namespace) -> tuple[pl.DataFrame, np.ndarray | None]:
    """Готовит базовую часть train и её веса один раз (без повторного пересчёта на каждом фолде CV)."""
    tr = frames["train"]
    w = None
    if a.pos_mode == "topn":
        tr = tr.filter(pl.col("pos").is_null() | (pl.col("pos") < a.topn))
    elif a.pos_mode in ("ipw", "ipw_all"):
        ctr = position_ctr(frames["train"]["pos"].to_numpy(), frames["train"]["target"].to_numpy())
        pos_only = a.pos_mode == "ipw"
        w = ipw_weights(
            tr["pos"].to_numpy(),
            tr["target"].to_numpy(),
            ctr,
            a.ipw_max,
            positives_only=pos_only,
            normalize_mean=not pos_only,
        )
    tr = apply_pos(tr, a, True)
    decay = getattr(a, "time_decay", 0.0)
    if decay > 0.0 and "timestamp" in tr.columns and tr.height:
        ts = tr["timestamp"].to_numpy().astype(np.float64)
        age_days = np.maximum(0.0, (ts.max() - ts) / 86400.0)
        tw = np.exp(-decay * age_days).astype(np.float32)
        w = tw if w is None else (w * tw).astype(np.float32)
    return tr, w


def _combine_train_val(
    base_tr: pl.DataFrame,
    base_w: np.ndarray | None,
    a: argparse.Namespace,
    val_rows: pl.DataFrame | None,
) -> tuple[pl.DataFrame, np.ndarray | None]:
    if val_rows is None or not val_rows.height:
        return base_tr, base_w
    v = _filter_zero_grad(val_rows) if getattr(a, "drop_zero_grad_feeds", False) else val_rows
    v = apply_pos(v, a, False)
    tr = pl.concat([base_tr, v.select(base_tr.columns)], how="vertical")
    w = base_w
    if base_w is not None or a.val_weight != 1.0:
        w0 = base_w if base_w is not None else np.ones(base_tr.height, dtype=np.float32)
        w = np.concatenate([w0, np.full(v.height, a.val_weight, dtype=np.float32)])
    return tr, w


def train_frame(
    frames: dict[str, pl.DataFrame], a: argparse.Namespace, val_rows: pl.DataFrame | None
) -> tuple[pl.DataFrame, np.ndarray | None]:
    """Совместимый интерфейс (используется tune.py и holdout/final)."""
    base_tr, base_w = _prepare_base_train(frames, a)
    return _combine_train_val(base_tr, base_w, a, val_rows)


def make_ranker(a: argparse.Namespace, feats: list[str], iterations: int | None = None) -> Ranker:
    feats_full = feats + (["pos"] if a.pos_mode == "feature" else [])
    extra = json.loads(Path(a.model_params).read_text()) if a.model_params else None
    return Ranker(a.kind, feats_full, a.loss, iterations or a.iterations, a.lr, a.depth, a.seed, a.task_type, extra)


def split_metrics(va: pl.DataFrame, score: np.ndarray, pub: np.ndarray, priv: np.ndarray) -> dict:
    f, t = va["feed_id"].to_numpy(), va["target"].to_numpy()
    out = {}
    for name, ids in (("all", None), ("pub", pub), ("priv", priv)):
        m = np.ones(len(f), bool) if ids is None else mask_feeds(f, ids)
        if m.any():
            for k, v in all_variants(f[m], t[m], score[m]).items():
                out[f"{name}_{k}"] = v
    return out


def append_log(a: argparse.Namespace, res: dict) -> None:
    log = Path(__file__).resolve().parents[2] / "experiments" / "log.md"
    if not log.exists():
        return
    desc = (
        f"{a.kind}/{a.loss or 'default'} pos={a.pos_mode} mode={a.mode} extra={','.join(a.extra) or '-'} "
        f"depth={a.depth} lr={a.lr} {a.note}"
    ).strip()
    v = "/".join(f"{res.get(k, float('nan')):.4f}" for k in ("priv_ndcg_full", "priv_ndcg_idcg30", "priv_ndcg@30"))
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"| {dt.date.today()} | danya | - | {desc} | priv: {v} | - | out={a.out} |\n")


def run(a: argparse.Namespace) -> dict:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    need_test = (not a.no_test) or a.mode == "final"
    frames, feats = load_all(a, need_test)
    model_feats = feats + (["pos"] if a.pos_mode == "feature" else [])
    va_raw = frames["val"]
    pub, priv = pseudo_split(va_raw["feed_id"].to_numpy(), va_raw["timestamp"].to_numpy())
    iters = a.iterations
    if a.iterations_from:
        iters = int(json.loads(Path(a.iterations_from).read_text())["best_iteration"])
    res: dict = {"args": vars(a), "n_features": len(model_feats)}
    test_scores: list[np.ndarray] = []

    te_keys: pl.DataFrame | None = None
    te_feed_np: np.ndarray | None = None
    te_prep = None
    if need_test and "test" in frames:
        te_df = apply_pos(frames.pop("test"), a, False)
        te_keys = te_df.select(pl.col("feed_id").cast(pl.Int32), pl.col("item_id").cast(pl.Int32))
        te_feed_np = te_keys["feed_id"].to_numpy()
        te_prep = prepare_predict_input(a.kind, te_df, model_feats)
        del te_df
        gc.collect()

    if a.mode == "holdout":
        tr, w = train_frame(frames, a, None)
        va = apply_pos(va_raw, a, False)
        es = va.filter(pl.col("feed_id").is_in(pub))
        model = make_ranker(a, feats, iters).fit(tr, w, es if es.height else va)
        score = model.predict(va)
        res.update(split_metrics(va, score, pub, priv))
        res["best_iteration"] = model.best_iteration
        try:
            res["feature_importance"] = model.importance()
            model.save(out / "model")
        except Exception as e:
            res["feature_importance_error"] = repr(e)
        val_pred = va.select("feed_id", "item_id", "target").with_columns(pl.Series("score", score))
        if te_prep is not None:
            test_scores.append(model.predict_prepared(te_prep))

    elif a.mode == "cv":
        uf, fold = time_folds(va_raw["feed_id"].to_numpy(), va_raw["timestamp"].to_numpy(), a.folds)
        fold_of = dict(zip(uf.tolist(), fold.tolist()))
        row_fold = np.array([fold_of[x] for x in va_raw["feed_id"].to_numpy()])
        oof = np.zeros(va_raw.height)
        base_tr, base_w = _prepare_base_train(frames, a)
        frames.pop("train", None)
        gc.collect()
        for k in range(a.folds):
            t_fold = time.perf_counter()
            held = row_fold == k
            tr, w = _combine_train_val(base_tr, base_w, a, va_raw.filter(pl.Series(~held)))
            model = make_ranker(a, feats, iters).fit(tr, w)
            del tr, w
            gc.collect()
            oof[held] = model.predict(apply_pos(va_raw.filter(pl.Series(held)), a, False))
            if te_prep is not None and te_feed_np is not None:
                test_scores.append(feed_zscore(te_feed_np, model.predict_prepared(te_prep)))
            del model
            gc.collect()
            print(f"[cv] fold {k + 1}/{a.folds} done in {time.perf_counter() - t_fold:.1f}s", flush=True)
        del base_tr, base_w
        va = apply_pos(va_raw, a, False)
        res.update(split_metrics(va, oof, pub, priv))
        val_pred = va.select("feed_id", "item_id", "target").with_columns(pl.Series("score", oof))

    else:  # final
        tr, w = train_frame(frames, a, va_raw)
        frames.pop("train", None)
        gc.collect()
        model = make_ranker(a, feats, iters).fit(tr, w)
        if te_prep is not None:
            test_scores.append(model.predict_prepared(te_prep))
        val_pred = None

    if need_test and test_scores and te_keys is not None:
        s = np.mean(test_scores, axis=0).astype(np.float32)
        sub = te_keys.with_columns(pl.Series("score", s))
        sub.write_parquet(out / "test_pred.parquet")
    if val_pred is not None:
        val_pred.write_parquet(out / "val_pred.parquet")
    (out / "metrics.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    if a.mode != "final":
        if a.log:
            append_log(a, res)
        print({k: round(v, 4) for k, v in res.items() if k.endswith(("ndcg_full", "ndcg@30"))})
    return res


if __name__ == "__main__":
    run(get_args())
