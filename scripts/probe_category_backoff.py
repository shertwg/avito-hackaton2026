"""Test a fixed category co-history adjustment on candidates without exact item matches."""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from scripts.run_joint_oof import digest
from src.common.cv import pseudo_split
from src.common.data import load_labels
from src.common.metric import per_feed_ndcg
from src.common.preparation import fingerprint, write_json
from src.common.scores import feed_zscore


KEY = ['feed_id', 'item_id']
WEIGHTS = [0., .025, .05, .1, .2]


def read_signal(directory, name, sources):
    manifest = json.loads((directory / 'manifest.json').read_bytes())
    if not manifest['complete'] or set(manifest['excluded_train_sources']) != set(sources):
        raise ValueError(f'incompatible donor cache: {directory}')
    parts = []
    for part in manifest['splits']['val']['parts'].values():
        path = directory / part['path']
        if fingerprint(path) != part['fingerprint']:
            raise ValueError(f'feature file changed: {path}')
        parts.append(pl.read_parquet(path, columns=[*KEY, name]))
    return pl.concat(parts)


def probe(data, model, item, category):
    report = json.loads((model / 'run.json').read_bytes())
    score_path = model / 'val_pred.parquet'
    if not report['complete'] or report['prediction_sha256'][score_path.name] != digest(score_path):
        raise ValueError('incomplete or changed OOF predictions')
    reference = load_labels(data, 'val')
    public, _ = pseudo_split(reference['feed_id'].to_numpy(), reference['timestamp'].to_numpy())
    reference = reference.filter(pl.col('feed_id').is_in(public))
    pred = pl.read_parquet(score_path).rename({'target': 'model_target'})
    sources = report['identity']['train_sources']
    for values in (pred, read_signal(item, 'i2i_max_50', sources),
                   read_signal(category, 'cc_sum_50', sources)):
        reference = reference.join(values, on=KEY, how='left', validate='1:1')
    if reference.null_count().to_numpy().any() or not reference['target'].equals(reference['model_target']):
        raise ValueError('missing keys, signals or inconsistent targets')
    feed = reference['feed_id'].to_numpy()
    target = reference['target'].to_numpy()
    early, late = pseudo_split(feed, reference['timestamp'].to_numpy(), frac=.5)
    raw = reference['score'].to_numpy()
    base = feed_zscore(feed, raw)
    missing = reference['i2i_max_50'].to_numpy() == 0
    signal = np.zeros(len(feed))
    # Center only among unsupported candidates so the adjustment has zero mean there.
    if missing.any():
        signal[missing] = feed_zscore(feed[missing], reference['cc_sum_50'].to_numpy()[missing])

    def metric(feeds, weight):
        mask = np.isin(feed, feeds)
        scores = raw[mask] if weight == 0 else base[mask] + weight * signal[mask]
        return per_feed_ndcg(feed[mask], target[mask], scores, cutoff=30)[1]

    grid = [float(metric(early, weight).mean()) for weight in WEIGHTS]
    weight = WEIGHTS[int(np.argmax(grid))]
    baseline = metric(late, 0)
    candidate = metric(late, weight)
    delta = candidate - baseline
    rng = np.random.default_rng(42)
    draws = [rng.choice(delta, len(delta)).mean() for _ in range(1000)]
    return dict(model=model.name, val_prediction_sha256=report['prediction_sha256'][score_path.name],
                scope='earliest and latest halves of pseudo-public; pseudo-private not scored',
                category_signal='within-feed z-score of cc_sum_50 among rows with i2i_max_50 == 0',
                weights=WEIGHTS, selection_ndcg30=grid, selected_weight=weight,
                selection_feeds=len(early), evaluation_feeds=len(late),
                evaluation_base=float(baseline.mean()), evaluation_adjusted=float(candidate.mean()),
                evaluation_delta=float(delta.mean()), paired_ci95=np.quantile(draws, [.025, .975]).tolist(),
                evaluation_feeds_up=int((delta > 0).sum()), evaluation_feeds_down=int((delta < 0).sum()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('competition'))
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--item', type=Path, default=Path('data/features/item2item-008'))
    parser.add_argument('--category', type=Path, default=Path('data/features/category-cooc-008'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = probe(args.data, args.model, args.item, args.category)
    write_json(args.out, result)
    print(json.dumps(result, indent=2))
