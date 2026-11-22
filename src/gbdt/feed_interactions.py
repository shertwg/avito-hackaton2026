"""Внутриленточные нормировки, ранги и взаимодействия сильнейших признаков (без утечек между лентами)."""
from __future__ import annotations

import polars as pl

NORM_COLS = [
    "i2i_sum_50",
    "i2i_sum_10",
    "i2i_max_50",
    "wi2i_all_mean_10",
    "wi2i_all_max_10",
    "wi2i_evt2_mean_50",
    "wi2i_all_mean_50",
    "rc_cos_profile_10",
    "rc_cos_profile_50",
    "is_item_history_evt1",
    "is_item_ctr",
    "item_train_age_days",
    "cos_hist_max_10",
    "cos_hist_mean_50",
    "cos_user",
    "ci2i_lift_mean_10",
    "ci2i_lift_max_10",
    "ci2i_lift_max_50",
    "cat_ci2i_lift_mean_10",
    "cat_ci2i_lift_max_10",
    "cat_ci2i_src_lift_mean_50",
    "cat_item_lift_max_50",
    "coclick_max_50",
    "us_donor_user_cat_lift",
    "us_donor_user_cat_clicks_log",
    "us_cat_prior_impr_share",
    "us_cat_recent3_impr_share",
]

RANK_COLS = [
    "i2i_sum_50",
    "i2i_sum_10",
    "i2i_max_50",
    "wi2i_all_mean_10",
    "wi2i_all_max_10",
    "wi2i_evt2_mean_50",
    "rc_cos_profile_10",
    "is_item_history_evt1",
    "item_train_age_days",
    "cos_hist_mean_50",
    "ci2i_lift_mean_10",
    "cat_ci2i_lift_mean_10",
    "cat_ci2i_src_lift_mean_50",
    "cat_item_lift_max_50",
    "us_donor_user_cat_lift",
    "us_donor_user_cat_clicks_log",
    "us_cat_recent3_impr_share",
]


def add_feed_interactions(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """Добавляет z-score, dmax, нормированные ранги внутри feed_id и доменные взаимодействия."""
    exprs: list[pl.Expr] = []
    cols: list[str] = []
    feed_n = pl.len().over("feed_id").cast(pl.Float32)

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

    for c in RANK_COLS:
        if c not in df.columns:
            continue
        r_name = f"fi_{c}_rank"
        exprs.append(
            (pl.col(c).cast(pl.Float32).rank(method="average").over("feed_id") / feed_n)
            .fill_nan(0.0).fill_null(0.0).cast(pl.Float32).alias(r_name)
        )
        cols.append(r_name)

    pairs = [
        ("fi_cos_x_ctr", "cos_user", "is_item_ctr", lambda a, b: a * b),
        ("fi_i2i50_x_cos", "i2i_sum_50", "cos_user", lambda a, b: a * b),
        ("fi_ctr_per_age", "is_item_ctr", "item_train_age_days", lambda a, b: a / (b + pl.lit(1.0, dtype=pl.Float32))),
        ("fi_i2i_ratio_10_50", "i2i_sum_10", "i2i_sum_50", lambda a, b: a / (b + pl.lit(1e-3, dtype=pl.Float32))),
        ("fi_i2i50_x_ctr", "i2i_sum_50", "is_item_ctr", lambda a, b: a * b),
        ("fi_i2i10_x_cos10", "i2i_sum_10", "cos_hist_max_10", lambda a, b: a * b),
        ("fi_evt1_x_cos50", "is_item_history_evt1", "cos_hist_mean_50", lambda a, b: a * b),
        ("fi_i2i_max_ratio", "i2i_max_10", "i2i_max_50", lambda a, b: a / (b + pl.lit(1e-3, dtype=pl.Float32))),
        ("fi_f3z_x_cos10", "f3_z", "cos_hist_max_10", lambda a, b: a * b),
        ("fi_f1z_x_i2i50", "f1_z", "i2i_sum_50", lambda a, b: a * b),
        ("fi_wi2i10_x_cos", "wi2i_all_mean_10", "cos_user", lambda a, b: a * b),
        ("fi_wi2i10_x_ctr", "wi2i_all_mean_10", "is_item_ctr", lambda a, b: a * b),
        ("fi_wi2i10_x_cos10", "wi2i_all_max_10", "cos_hist_max_10", lambda a, b: a * b),
        ("fi_wi2i_ratio_10_50", "wi2i_all_mean_10", "wi2i_all_mean_50", lambda a, b: a / (b + pl.lit(1e-4, dtype=pl.Float32))),
        ("fi_wi2i_evt2_ratio", "wi2i_evt2_mean_50", "wi2i_all_mean_50", lambda a, b: a / (b + pl.lit(1e-4, dtype=pl.Float32))),
        ("fi_ci2i10_x_cos", "ci2i_lift_mean_10", "cos_user", lambda a, b: a * b),
        ("fi_ci2i10_x_ctr", "ci2i_lift_mean_10", "is_item_ctr", lambda a, b: a * b),
        ("fi_catci2i10_x_cos", "cat_ci2i_lift_mean_10", "cos_user", lambda a, b: a * b),
        ("fi_catci2i10_x_wi2i10", "cat_ci2i_lift_mean_10", "wi2i_all_mean_10", lambda a, b: a * b),
        ("fi_catci2i50_x_cos", "cat_ci2i_src_lift_mean_50", "cos_user", lambda a, b: a * b),
        ("fi_us_cat_fatigue_delta", "us_cat_recent3_impr_share", "us_cat_prior_impr_share", lambda a, b: a - b),
        ("fi_us_lift_x_cos", "us_donor_user_cat_lift", "cos_user", lambda a, b: a * b),
        ("fi_us_lift_x_ctr", "us_donor_user_cat_lift", "is_item_ctr", lambda a, b: a * b),
        ("fi_us_cat_clicks_x_cos", "us_donor_user_cat_clicks_log", "cos_user", lambda a, b: a * b),
        ("fi_pt_x_cos", "pt_gain", "cos_user", lambda a, b: a * b),
        ("fi_pt_x_ctr", "pt_gain", "is_item_ctr", lambda a, b: a * b),
    ]
    for name, c1, c2, fn in pairs:
        if c1 in df.columns and c2 in df.columns:
            exprs.append(
                fn(pl.col(c1).cast(pl.Float32), pl.col(c2).cast(pl.Float32))
                .fill_nan(0.0).fill_null(0.0).cast(pl.Float32).alias(name)
            )
            cols.append(name)

    if "category_id" in df.columns:
        cat_n = pl.len().over(["feed_id", "category_id"]).cast(pl.Float32)
        exprs.append(cat_n.alias("fi_incat_count"))
        exprs.append((cat_n / feed_n).fill_nan(0.0).fill_null(0.0).cast(pl.Float32).alias("fi_incat_share"))
        exprs.append(
            (pl.col("category_id").n_unique().over("feed_id").cast(pl.Float32) / feed_n)
            .fill_nan(0.0)
            .fill_null(0.0)
            .cast(pl.Float32)
            .alias("fi_feed_unique_cats_ratio")
        )
        cols.extend(["fi_incat_count", "fi_incat_share", "fi_feed_unique_cats_ratio"])
        for c in ("cos_user", "is_item_ctr", "wi2i_all_mean_10", "cat_item_lift_max_50", "ci2i_lift_mean_10"):
            if c in df.columns:
                col_f = pl.col(c).cast(pl.Float32)
                cname = f"fi_incat_{c}_dmax"
                exprs.append((col_f - col_f.max().over(["feed_id", "category_id"])).fill_nan(0.0).fill_null(0.0).cast(pl.Float32).alias(cname))
                cols.append(cname)
        for c in ("f1", "f5"):
            if c in df.columns:
                col_f = pl.col(c).cast(pl.Float32)
                mean_c = col_f.mean().over(["feed_id", "category_id"])
                std_c = col_f.std(ddof=0).over(["feed_id", "category_id"]).fill_nan(0.0).fill_null(0.0)
                z_name = f"fi_incat_{c}_z"
                dmax_name = f"fi_incat_{c}_dmax"
                exprs.append(((col_f - mean_c) / (std_c + 1e-6)).fill_nan(0.0).fill_null(0.0).cast(pl.Float32).alias(z_name))
                exprs.append((col_f - col_f.max().over(["feed_id", "category_id"])).fill_nan(0.0).fill_null(0.0).cast(pl.Float32).alias(dmax_name))
                cols.extend([z_name, dmax_name])
        for c in ("f1", "f5"):
            if c in df.columns:
                col_f = pl.col(c).cast(pl.Float32)
                r_name = f"fi_incat_{c}_rank"
                exprs.append((col_f.rank(method="average").over(["feed_id", "category_id"]).cast(pl.Float32) / cat_n).fill_nan(0.0).fill_null(0.0).cast(pl.Float32).alias(r_name))
                cols.append(r_name)

    for c in ["i2i_sum_50", "is_item_history_evt1", "item_train_age_days"]:
        if c in df.columns:
            lname = f"fi_log_{c}"
            exprs.append(
                pl.col(c).cast(pl.Float32).clip(lower_bound=0.0).log1p()
                .fill_nan(0.0).fill_null(0.0).cast(pl.Float32).alias(lname)
            )
            cols.append(lname)

    if exprs:
        df = df.with_columns(exprs)
    return df, cols
