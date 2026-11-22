"""Работа со скорами: нормировка внутри ленты (скоры разных моделей несопоставимы по шкале)."""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy.special import ndtri


def feed_zscore(feed_id: np.ndarray, score: np.ndarray) -> np.ndarray:
    d = pl.DataFrame({"f": feed_id, "s": np.asarray(score, dtype=np.float64)})
    out = d.with_columns(
        ((pl.col("s") - pl.col("s").mean().over("f")) / (pl.col("s").std().over("f") + 1e-9)).alias("z")
    )
    return out["z"].fill_null(0.0).to_numpy()


def feed_rank01(feed_id: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Ранг внутри ленты в (0, 1], 1 = лучший."""
    d = pl.DataFrame({"f": feed_id, "s": np.asarray(score, dtype=np.float64)})
    out = d.with_columns((pl.col("s").rank("average").over("f") / pl.len().over("f")).alias("r"))
    return out["r"].to_numpy()


def feed_hybrid_rankgauss(feed_id: np.ndarray, score: np.ndarray, rank_alpha: float = 0.25) -> np.ndarray:
    """Гибрид Z-score и RankGauss внутри ленты: (1 - alpha) * z + alpha * z(ndtri(clip(r01, 1e-4, 1 - 1e-4)))."""
    z = feed_zscore(feed_id, score)
    if rank_alpha <= 0.0:
        return z.astype(np.float32)
    r01 = feed_rank01(feed_id, score)
    rg = ndtri(np.clip(r01, 1e-4, 1.0 - 1e-4)).astype(np.float32)
    rg_z = feed_zscore(feed_id, rg)
    return ((1.0 - rank_alpha) * z + rank_alpha * rg_z).astype(np.float32)
