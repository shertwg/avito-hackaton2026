"""Внутриленточные нормировки и взаимодействия сильнейших признаков (без утечек между лентами)."""
from __future__ import annotations

import polars as pl

NORM_COLS = [
    "i2i_sum_50",
    "i2i_sum_10",
    "i2i_max_50",
    "is_item_history_evt1",
    "is_item_ctr",
    "item_train_age_days",
    "cos_hist_max_10",
    "cos_hist_mean_50",
    "cos_user",
]


def add_feed_interactions(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """Добавляет z-score и отклонение от максимума внутри feed_id, а также ключевые взаимодействия."""
    exprs: list[pl.Expr] = []
    cols: list[str] = []

    for c in NORM_COLS:
        if c not in df.columns:
            continue
        col_f = pl.col(c).cast(pl.Float32)
        z_name = f"fi_{c}_z"
        d_name = f"fi_{c}_dmax"
        std = col_f.std().over("feed_id").fill_null(0.0)
        mean = col_f.mean().over("feed_id")
        mx = col_f.max().over("feed_id")
        exprs.append(
            ((col_f - mean) / (std + pl.lit(1e-6, dtype=pl.Float32)))
            .fill_nan(0.0).fill_null(0.0).cast(pl.Float32).alias(z_name)
        )
        exprs.append(
            (col_f - mx).fill_nan(0.0).fill_null(0.0).cast(pl.Float32).alias(d_name)
        )
        cols.extend([z_name, d_name])

    pairs = [
        ("fi_cos_x_ctr", "cos_user", "is_item_ctr", lambda a, b: a * b),
        ("fi_i2i50_x_cos", "i2i_sum_50", "cos_user", lambda a, b: a * b),
        ("fi_ctr_per_age", "is_item_ctr", "item_train_age_days", lambda a, b: a / (b + pl.lit(1.0, dtype=pl.Float32))),
        ("fi_i2i_ratio_10_50", "i2i_sum_10", "i2i_sum_50", lambda a, b: a / (b + pl.lit(1e-3, dtype=pl.Float32))),
    ]
    for name, c1, c2, fn in pairs:
        if c1 in df.columns and c2 in df.columns:
            exprs.append(
                fn(pl.col(c1).cast(pl.Float32), pl.col(c2).cast(pl.Float32))
                .fill_nan(0.0).fill_null(0.0).cast(pl.Float32).alias(name)
            )
            cols.append(name)

    if exprs:
        df = df.with_columns(exprs)
    return df, cols
