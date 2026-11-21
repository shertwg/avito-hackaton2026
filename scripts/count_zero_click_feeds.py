"""Count exactly which raw ranking groups an empty-target filter would remove."""

import argparse
from collections import Counter
from pathlib import Path

import polars as pl

from src.common.data import list_parts
from src.common.preparation import write_json


def count_groups(raw):
    result = {}
    for split in ("train", "val"):
        total = Counter()
        parts = list_parts(raw, split)
        for path in parts:
            df = pl.read_parquet(path, columns=["feed_id", "target"])
            if df["feed_id"].is_duplicated().any() or df["target"].null_count():
                raise ValueError(f"invalid feed or target in {path.name}")
            stat = df.select(
                pl.col("target").list.len().alias("rows"),
                pl.col("target").list.eval(pl.element().cast(pl.Int64)).list.sum().alias("pos"),
                pl.col("target")
                .list.eval(pl.element().is_in([0, 1]).fill_null(False))
                .list.all()
                .alias("valid"),
            )
            if not stat["valid"].all() or stat["rows"].min() < 1:
                raise ValueError(f"empty feed or non-binary target in {path.name}")
            no_click = stat["pos"] == 0
            all_click = stat["pos"] == stat["rows"]
            total.update(
                feeds=stat.height,
                candidates=int(stat["rows"].sum()),
                clicks=int(stat["pos"].sum()),
                zero_click_feeds=int(no_click.sum()),
                zero_click_candidates=int(stat["rows"].filter(no_click).sum()),
                all_click_feeds=int(all_click.sum()),
                all_click_candidates=int(stat["rows"].filter(all_click).sum()),
            )
        result[split] = dict(shards=len(parts), **total)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("competition"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = count_groups(args.raw)
    write_json(args.out, report)
    print(report)
