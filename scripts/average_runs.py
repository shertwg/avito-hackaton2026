"""Сид-ансамбль: среднее z-скоров (внутри ленты) нескольких запусков одной модели.

Результат — обычная папка запуска (val_pred.parquet с target, test_pred.parquet), её можно
передавать в src.ensemble.blend и scripts.paired_compare. Веса не подбираются, поэтому
усреднение не смотрит ни на pseudo-public, ни на pseudo-private.

    uv run --locked python -m scripts.average_runs --runs runs/cb8-cv runs/cb8-cv-s1 runs/cb8-cv-s2 --out runs/cb8-cv-seeds
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from src.common.scores import feed_zscore

KEYS = ['feed_id', 'item_id']


def average(runs: list[Path], which: str) -> pl.DataFrame:
    base = None
    zs = []
    for run in runs:
        df = pl.read_parquet(run / f'{which}_pred.parquet').sort(KEYS)
        if base is None:
            base = df.drop('score')
        elif not df.select(KEYS).equals(base.select(KEYS)):
            raise SystemExit(f'{run}: {which} rows differ from {runs[0]}')
        zs.append(feed_zscore(df['feed_id'].to_numpy(), df['score'].to_numpy()))
    return base.with_columns(pl.Series('score', np.mean(zs, axis=0).astype(np.float32)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--runs', type=Path, nargs='+', required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for which in ('val', 'test'):
        df = average(a.runs, which)
        if which == 'test':
            df = df.select(pl.col('feed_id').cast(pl.Int32), pl.col('item_id').cast(pl.Int32), 'score')
        df.write_parquet(a.out / f'{which}_pred.parquet')
    (a.out / 'metrics.json').write_text(json.dumps({'averaged_runs': [str(r) for r in a.runs]}, indent=2))
    print(f'averaged {len(a.runs)} runs -> {a.out}')


if __name__ == '__main__':
    main()
