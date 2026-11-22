"""Единая обёртка над CatBoost и LightGBM для ранжирования (группа = лента)."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostRanker, Pool

CAT = "category_id"
_POINTWISE_LGB = {"binary", "cross_entropy", "regression", "regression_l1", "huber"}


def _x(df: pl.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Zero-copy поколоночная сборка Pandas DataFrame (экономит ~16 ГБ RAM против df.to_pandas())."""
    cols: dict[str, np.ndarray] = {}
    for c in feats:
        arr = df[c].to_numpy()
        if c == CAT:
            cols[c] = arr.astype(np.int32, copy=False)
        else:
            cols[c] = arr.astype(np.float32, copy=False)
    return pd.DataFrame(cols, copy=False)


def _x_f32(df: pl.DataFrame, feats: list[str]) -> np.ndarray:
    """Прямая запись колонок в единый float32-буфер без промежуточной копии Polars DataFrame."""
    out = np.empty((df.height, len(feats)), dtype=np.float32)
    for j, c in enumerate(feats):
        out[:, j] = df[c].to_numpy()
    return out


def _group_sizes(feed_id: np.ndarray) -> np.ndarray:
    """Размеры непрерывных групп лент за O(N) (с резервным путём при неотсортированном входе)."""
    if len(feed_id) == 0:
        return np.empty(0, dtype=np.int32)
    if len(feed_id) == 1 or np.all(feed_id[1:] >= feed_id[:-1]):
        bounds = np.flatnonzero(feed_id[1:] != feed_id[:-1]) + 1
        return np.diff(np.r_[0, bounds, len(feed_id)])
    _, counts = np.unique(feed_id, return_counts=True)
    return counts


def prepare_predict_input(kind: str, df: pl.DataFrame, feats: list[str]):
    """Готовит неизменяемый вход для многократного предсказания (например, test на 4 фолдах CV)."""
    if kind == "catboost":
        cat_cols = [CAT] if CAT in feats else None
        return Pool(_x(df, feats), group_id=df["feed_id"].to_numpy(), cat_features=cat_cols)
    return _x_f32(df, feats)


class Ranker:
    def __init__(
        self,
        kind: str,
        feats: list[str],
        loss: str | None = None,
        iterations: int = 2000,
        lr: float = 0.05,
        depth: int = 8,
        seed: int = 0,
        task_type: str = "CPU",
        extra: dict | None = None,
    ):
        self.kind, self.feats, self.seed = kind, feats, seed
        self.iterations, self.lr, self.depth = iterations, lr, depth
        self.loss = loss or ("YetiRank" if kind == "catboost" else "lambdarank")
        self.task_type, self.extra = task_type, extra or {}
        self.model = None
        self.best_iteration: int | None = None

    # ---- обучение -------------------------------------------------------------------------
    def fit(
        self,
        tr: pl.DataFrame,
        weight: np.ndarray | None = None,
        va: pl.DataFrame | None = None,
        es_rounds: int = 100,
    ) -> "Ranker":
        if weight is not None and self.kind == "catboost" and self.loss.startswith(("YetiRank", "PairLogit")):
            # CatBoost для парных лоссов только пишет предупреждение и ИГНОРИРУЕТ веса объектов
            raise ValueError(
                f"CatBoost {self.loss} не поддерживает веса объектов (IPW/val-weight). "
                "Используйте --kind lightgbm или --loss LambdaMart / QueryRMSE / QuerySoftMax."
            )
        return (
            self._fit_cb(tr, weight, va, es_rounds)
            if self.kind == "catboost"
            else self._fit_lgb(tr, weight, va, es_rounds)
        )

    def _fit_cb(self, tr, weight, va, es_rounds):
        extra = dict(self.extra)
        loss_fn = self.loss
        params = dict(
            iterations=self.iterations,
            learning_rate=self.lr,
            depth=self.depth,
            random_seed=self.seed,
            task_type=self.task_type,
            verbose=200,
            allow_writing_files=False,
        )
        if self.task_type == "GPU":
            extra.setdefault("border_count", 254)
            extra.setdefault("one_hot_max_size", 100)
            if "skip_train" not in loss_fn:
                loss_fn += (";" if ":" in loss_fn else ":") + "hints=skip_train~true"
            if va is not None:
                params["eval_metric"] = "NDCG:top=30;type=Base;hints=skip_train~true"
        else:
            extra.setdefault("thread_count", os.cpu_count() or 8)
            params["eval_metric"] = "NDCG:top=30;type=Base"
        params["loss_function"] = loss_fn
        params.update(extra)
        if va is not None:
            params.update(od_type="Iter", od_wait=es_rounds)
        m = CatBoostRanker(**params)
        cat_cols = [CAT] if CAT in self.feats else None
        pool = Pool(
            _x(tr, self.feats),
            label=tr["target"].to_numpy(),
            group_id=tr["feed_id"].to_numpy(),
            cat_features=cat_cols,
            weight=weight,
        )
        ev = None
        if va is not None:
            ev = Pool(
                _x(va, self.feats),
                label=va["target"].to_numpy(),
                group_id=va["feed_id"].to_numpy(),
                cat_features=cat_cols,
            )
        m.fit(pool, eval_set=ev, use_best_model=va is not None)
        self.model = m
        self.best_iteration = int(m.get_best_iteration()) + 1 if va is not None else self.iterations
        return self

    def _fit_lgb(self, tr, weight, va, es_rounds):
        import lightgbm as lgb

        extra = dict(self.extra)
        extra.setdefault("num_threads", os.cpu_count() or 8)
        extra.setdefault("force_col_wise", True)
        need_groups = (self.loss not in _POINTWISE_LGB) or (va is not None)
        params = dict(
            objective=self.loss,
            learning_rate=self.lr,
            num_leaves=2 ** min(self.depth, 8) - 1,
            max_depth=-1,
            metric="ndcg" if need_groups else "None",
            eval_at=[30] if need_groups else [],
            seed=self.seed,
            verbose=-1,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            **extra,
        )
        cat_idx = [self.feats.index(CAT)] if CAT in self.feats else []
        tr_groups = _group_sizes(tr["feed_id"].to_numpy()) if need_groups else None
        dtr = lgb.Dataset(
            _x_f32(tr, self.feats),
            label=tr["target"].to_numpy(),
            weight=weight,
            group=tr_groups,
            feature_name=self.feats,
            categorical_feature=cat_idx,
            free_raw_data=True,
        )
        cbs = [lgb.log_evaluation(200)]
        valid = []
        if va is not None:
            valid = [
                lgb.Dataset(
                    _x_f32(va, self.feats),
                    label=va["target"].to_numpy(),
                    group=_group_sizes(va["feed_id"].to_numpy()),
                    feature_name=self.feats,
                    categorical_feature=cat_idx,
                    reference=dtr,
                    free_raw_data=True,
                )
            ]
            cbs.append(lgb.early_stopping(es_rounds, verbose=False))
        self.model = lgb.train(params, dtr, num_boost_round=self.iterations, valid_sets=valid, callbacks=cbs)
        self.best_iteration = int(self.model.best_iteration or self.iterations) if va is not None else self.iterations
        return self

    # ---- важность признаков и сохранение ---------------------------------------------------
    def importance(self) -> dict[str, float]:
        """Важность по модели без данных: CatBoost PredictionValuesChange, LightGBM gain. Сумма = 100."""
        if self.kind == "catboost":
            vals = np.asarray(self.model.get_feature_importance(type="PredictionValuesChange"), dtype=float)
        else:
            vals = np.asarray(self.model.feature_importance(importance_type="gain"), dtype=float)
        total = vals.sum() or 1.0
        pairs = sorted(zip(self.feats, (100.0 * vals / total).tolist()), key=lambda p: -p[1])
        return {name: round(v, 4) for name, v in pairs}

    def save(self, path) -> None:
        if self.kind == "catboost":
            self.model.save_model(str(path) + ".cbm")
        else:
            self.model.save_model(str(path) + ".txt", num_iteration=self.best_iteration)

    # ---- предсказание ---------------------------------------------------------------------
    def predict_prepared(self, x_prep) -> np.ndarray:
        if self.kind == "catboost":
            return np.asarray(self.model.predict(x_prep), dtype=np.float64)
        return np.asarray(self.model.predict(x_prep, num_iteration=self.best_iteration), dtype=np.float64)

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        return self.predict_prepared(prepare_predict_input(self.kind, df, self.feats))
