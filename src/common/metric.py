"""NDCG по правилам соревнования Авито (векторизованный расчёт + быстрый OOF-оценщик).

Правила:
- внутри ленты сортируем по убыванию score; при равных score негативы раньше позитивов;
- лента без позитивов -> NDCG = 1;
- итог = среднее по лентам с равным весом.

Эталон competition/baseline/calculate_metrics.py считает full, @10 и @30.
При cutoff=k и DCG, и IDCG ограничены первыми k кандидатами:
- idcg_k=None   : IDCG по всем позитивам (классический NDCG без отсечения)
- idcg_k=30     : экспериментальный вариант из описания, не метрика эталона
- cutoff=30     : и DCG, и IDCG только по первым 30 позициям (NDCG@30)
"""
from __future__ import annotations

import numpy as np


def _discounts(n: int) -> np.ndarray:
    return 1.0 / np.log2(np.arange(2, n + 2, dtype=np.float64))


def _validate(target: np.ndarray, score: np.ndarray, cutoff: int | None, idcg_k: int | None) -> None:
    if target.ndim != 1 or score.ndim != 1 or target.shape != score.shape:
        raise ValueError("target and score must be one-dimensional arrays of equal length")
    if not len(target):
        raise ValueError("cannot evaluate an empty set of candidates")
    if not np.isin(target, [0, 1]).all():
        raise ValueError("target must contain only binary labels (0/1)")
    if not np.isfinite(score).all():
        raise ValueError("score must be finite")
    for name, value in (("cutoff", cutoff), ("idcg_k", idcg_k)):
        if value is not None and (not isinstance(value, (int, np.integer)) or value <= 0):
            raise ValueError(f"{name} must be a positive integer")
    if cutoff is not None and idcg_k is not None:
        raise ValueError("use either cutoff or idcg_k, not both")


def ndcg_one(
    target: np.ndarray,
    score: np.ndarray,
    cutoff: int | None = None,
    idcg_k: int | None = None,
) -> float:
    target = np.asarray(target, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    _validate(target, score, cutoff, idcg_k)
    n = len(target)
    if target.sum() == 0:
        return 1.0
    # тай-брейк: при равных score негатив выше позитива.
    # lexsort сортирует по последнему ключу первым: сначала -score, затем target (0 раньше 1)
    order = np.lexsort((target, -score))
    disc = _discounts(n)
    ranked = target[order]
    if cutoff is not None:
        dcg = float((ranked[:cutoff] * disc[:cutoff]).sum())
    else:
        dcg = float((ranked * disc).sum())
    ideal = np.sort(target)[::-1]
    k = cutoff if cutoff is not None else idcg_k
    if k is not None:
        idcg = float((ideal[:k] * disc[:k]).sum())
    else:
        idcg = float((ideal * disc).sum())
    return dcg / idcg if idcg > 0 else 1.0


class FastNDCGEvaluator:
    """Предкомпилированный векторизованный оценщик NDCG по фиксированным (feed_ids, target).

    Вместо сортировки всех 1.39M строк или 14 423 вызовов Python вычисляет точный ранг
    только для позитивных объектов внутри нетривиальных лент (где 0 < positives < len),
    разбивая их на компактные бакеты ширины (<=128 и >128) для минимального трафика кэша.
    Даёт побитово/численно эквивалентный результат (~1e-16) в 20-50 раз быстрее.
    """

    def __init__(
        self,
        feed_ids: np.ndarray,
        target: np.ndarray,
        cutoff: int | None = 30,
        idcg_k: int | None = None,
    ) -> None:
        feed_ids = np.asarray(feed_ids)
        target_arr = np.asarray(target)
        if target_arr.ndim != 1 or feed_ids.ndim != 1 or len(feed_ids) != len(target_arr):
            raise ValueError("feed_ids must be a one-dimensional array aligned with target and score")
        if not len(target_arr):
            raise ValueError("cannot evaluate an empty set of candidates")
        if not np.issubdtype(feed_ids.dtype, np.integer):
            raise ValueError("feed_ids must contain integer IDs")
        if not np.isin(target_arr, [0, 1]).all():
            raise ValueError("target must contain only binary labels (0/1)")
        for name, value in (("cutoff", cutoff), ("idcg_k", idcg_k)):
            if value is not None and (not isinstance(value, (int, np.integer)) or value <= 0):
                raise ValueError(f"{name} must be a positive integer")
        if cutoff is not None and idcg_k is not None:
            raise ValueError("use either cutoff or idcg_k, not both")

        self.n_rows = len(feed_ids)
        self.cutoff = int(cutoff) if cutoff is not None else None
        self.idcg_k = int(idcg_k) if idcg_k is not None else None

        # Проверяем, отсортирован ли уже массив по feed_id
        if self.n_rows <= 1 or np.all(feed_ids[1:] >= feed_ids[:-1]):
            self.order = None
            fs = feed_ids
            ts = target_arr.astype(np.int8, copy=False)
        else:
            self.order = np.argsort(feed_ids, kind="stable")
            fs = feed_ids[self.order]
            ts = target_arr[self.order].astype(np.int8, copy=False)

        bounds = np.flatnonzero(fs[1:] != fs[:-1]) + 1
        starts = np.r_[0, bounds]
        ends = np.r_[bounds, len(fs)]
        lengths = ends - starts
        self.n_feeds = len(starts)
        self.unique_feeds = fs[starts]

        max_len = int(lengths.max())
        max_disc = max_len if self.cutoff is None else min(max_len, self.cutoff)
        idcg_cap = self.cutoff if self.cutoff is not None else self.idcg_k
        max_idcg_len = max_len if idcg_cap is None else min(max_len, idcg_cap)

        self.disc = _discounts(max(max_disc, max_idcg_len, 1))
        disc_csum = np.r_[0.0, np.cumsum(self.disc)]

        feed_of_row = np.repeat(np.arange(self.n_feeds, dtype=np.int32), lengths)
        pos_rows_all = np.flatnonzero(ts == 1)
        pos_feeds_all = feed_of_row[pos_rows_all]
        pos_counts = np.bincount(pos_feeds_all, minlength=self.n_feeds)

        # Нетривиальные ленты: есть хотя бы 1 клик, и либо есть не-клик, либо включён idcg_k без отсечения DCG
        if self.idcg_k is None:
            nontrivial_feed = (pos_counts > 0) & (pos_counts < lengths)
        else:
            nontrivial_feed = pos_counts > 0
        self.trivial_one = ~nontrivial_feed

        k_eff = pos_counts if idcg_cap is None else np.minimum(pos_counts, idcg_cap)
        idcg = disc_csum[k_eff]
        self.inv_idcg = np.where(idcg > 0, 1.0 / np.maximum(idcg, 1e-30), 0.0)

        # Компактное подмножество строк только из нетривиальных лент (ускоряет X @ w в 1.5-2 раза)
        nt_row_mask = nontrivial_feed[feed_of_row]
        self.nt_rows = np.flatnonzero(nt_row_mask)
        if len(self.nt_rows):
            remap = np.full(self.n_rows, -1, dtype=np.int32)
            remap[self.nt_rows] = np.arange(len(self.nt_rows), dtype=np.int32)
            nt_feed_ids = np.flatnonzero(nontrivial_feed)
            nt_starts = remap[starts[nt_feed_ids]]
            nt_lengths = lengths[nt_feed_ids]

            keep_pos = nontrivial_feed[pos_feeds_all]
            pos_rows = remap[pos_rows_all[keep_pos]]
            pos_orig_feeds = pos_feeds_all[keep_pos]
            # Локальный индекс нетривиальной ленты для каждого позитива
            feed_to_nt = np.full(self.n_feeds, -1, dtype=np.int32)
            feed_to_nt[nt_feed_ids] = np.arange(len(nt_feed_ids), dtype=np.int32)
            pos_nt_idx = feed_to_nt[pos_orig_feeds]
            pos_lens = nt_lengths[pos_nt_idx]
            pos_starts = nt_starts[pos_nt_idx]
            ts_nt = ts[self.nt_rows]

            self.nt_feed_ids = nt_feed_ids
            self.n_nt_feeds = len(nt_feed_ids)
            self.buckets: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
            # Разбиваем на 2 бакета ширины (<=128 и >128), чтобы короткие ленты не читали 500 столбцов
            short_mask = pos_lens <= 128
            for bmask in (short_mask, ~short_mask):
                if not np.any(bmask):
                    continue
                b_pos_rows = pos_rows[bmask]
                b_pos_feeds = pos_orig_feeds[bmask]
                b_lens = pos_lens[bmask]
                b_starts = pos_starts[bmask]
                w = int(b_lens.max())
                cols = np.arange(w, dtype=np.int32)[None, :]
                valid = cols < b_lens[:, None]
                cand_idx = np.minimum(b_starts[:, None] + cols, len(self.nt_rows) - 1)
                cand_t = ts_nt[cand_idx]
                tie_before = valid & ((cand_t == 0) | ((cand_t == 1) & (cand_idx < b_pos_rows[:, None])))
                self.buckets.append((b_pos_rows, b_pos_feeds, cand_idx, valid, tie_before))
        else:
            self.nt_feed_ids = np.empty(0, dtype=np.int32)
            self.n_nt_feeds = 0
            self.buckets = []

    def compact_matrix(self, X: np.ndarray) -> np.ndarray:
        """Вырезает строки X только для нетривиальных лент (для сверхбыстрого поиска весов X_nt @ w)."""
        X_ord = X if self.order is None else X[self.order]
        return np.ascontiguousarray(X_ord[self.nt_rows], dtype=np.float32)

    def _per_feed_from_nt_scores(self, s_nt: np.ndarray) -> np.ndarray:
        out = np.ones(self.n_feeds, dtype=np.float64)
        if not self.n_nt_feeds:
            return out
        dcg = np.zeros(self.n_feeds, dtype=np.float64)
        cut = self.cutoff
        for b_pos_rows, b_pos_feeds, cand_idx, valid, tie_before in self.buckets:
            sc_cand = s_nt[cand_idx]
            sc_pos = s_nt[b_pos_rows][:, None]
            ranks = np.count_nonzero(valid & (sc_cand > sc_pos), axis=1) + np.count_nonzero(
                tie_before & (sc_cand == sc_pos), axis=1
            )
            if cut is not None:
                hit = ranks < cut
                if np.any(hit):
                    dcg += np.bincount(b_pos_feeds[hit], weights=self.disc[ranks[hit]], minlength=self.n_feeds)
            else:
                dcg += np.bincount(b_pos_feeds, weights=self.disc[ranks], minlength=self.n_feeds)
        out[self.nt_feed_ids] = dcg[self.nt_feed_ids] * self.inv_idcg[self.nt_feed_ids]
        return out

    def mean_from_compact(self, s_nt: np.ndarray) -> float:
        """Считает точный mean_ndcg по всем лентам из скоров только нетривиальных строк."""
        if not self.n_nt_feeds:
            return 1.0
        dcg = np.zeros(self.n_feeds, dtype=np.float64)
        cut = self.cutoff
        for b_pos_rows, b_pos_feeds, cand_idx, valid, tie_before in self.buckets:
            sc_cand = s_nt[cand_idx]
            sc_pos = s_nt[b_pos_rows][:, None]
            ranks = np.count_nonzero(valid & (sc_cand > sc_pos), axis=1) + np.count_nonzero(
                tie_before & (sc_cand == sc_pos), axis=1
            )
            if cut is not None:
                hit = ranks < cut
                if np.any(hit):
                    dcg += np.bincount(b_pos_feeds[hit], weights=self.disc[ranks[hit]], minlength=self.n_feeds)
            else:
                dcg += np.bincount(b_pos_feeds, weights=self.disc[ranks], minlength=self.n_feeds)
        nt_sum = float(np.dot(dcg[self.nt_feed_ids], self.inv_idcg[self.nt_feed_ids]))
        n_trivial = self.n_feeds - self.n_nt_feeds
        return (nt_sum + n_trivial) / self.n_feeds

    def per_feed(self, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        score_arr = np.asarray(score)
        if score_arr.ndim != 1 or len(score_arr) != self.n_rows:
            raise ValueError("target and score must be one-dimensional arrays of equal length")
        if not np.isfinite(score_arr).all():
            raise ValueError("score must be finite")
        s_ord = score_arr if self.order is None else score_arr[self.order]
        s_nt = s_ord[self.nt_rows]
        return self.unique_feeds, self._per_feed_from_nt_scores(s_nt)

    def mean(self, score: np.ndarray) -> float:
        score_arr = np.asarray(score)
        if score_arr.ndim != 1 or len(score_arr) != self.n_rows:
            raise ValueError("target and score must be one-dimensional arrays of equal length")
        if not np.isfinite(score_arr).all():
            raise ValueError("score must be finite")
        s_ord = score_arr if self.order is None else score_arr[self.order]
        return self.mean_from_compact(s_ord[self.nt_rows])


def per_feed_ndcg(
    feed_ids: np.ndarray,
    target: np.ndarray,
    score: np.ndarray,
    cutoff: int | None = None,
    idcg_k: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """NDCG каждой ленты: (отсортированные feed_id, значения). Нужно для бутстрепа и сравнения моделей."""
    feed_ids = np.asarray(feed_ids)
    target = np.asarray(target)
    score = np.asarray(score)
    _validate(target, score, cutoff, idcg_k)
    if feed_ids.ndim != 1 or len(feed_ids) != len(target):
        raise ValueError("feed_ids must be a one-dimensional array aligned with target and score")
    if not np.issubdtype(feed_ids.dtype, np.integer):
        raise ValueError("feed_ids must contain integer IDs")
    ev = FastNDCGEvaluator(feed_ids, target, cutoff=cutoff, idcg_k=idcg_k)
    return ev.per_feed(score)


def mean_ndcg(
    feed_ids: np.ndarray,
    target: np.ndarray,
    score: np.ndarray,
    cutoff: int | None = None,
    idcg_k: int | None = None,
) -> float:
    """Среднее NDCG по лентам. Массивы выровнены по строкам."""
    feed_ids = np.asarray(feed_ids)
    target = np.asarray(target, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    _validate(target, score, cutoff, idcg_k)
    if feed_ids.ndim != 1 or len(feed_ids) != len(target):
        raise ValueError("feed_ids must be a one-dimensional array aligned with target and score")
    if not np.issubdtype(feed_ids.dtype, np.integer):
        raise ValueError("feed_ids must contain integer IDs")
    ev = FastNDCGEvaluator(feed_ids, target, cutoff=cutoff, idcg_k=idcg_k)
    return ev.mean(score)


def all_variants(feed_ids, target, score) -> dict[str, float]:
    return {
        "ndcg_full": mean_ndcg(feed_ids, target, score),
        "ndcg_idcg30": mean_ndcg(feed_ids, target, score, idcg_k=30),
        "ndcg@10": mean_ndcg(feed_ids, target, score, cutoff=10),
        "ndcg@30": mean_ndcg(feed_ids, target, score, cutoff=30),
    }
