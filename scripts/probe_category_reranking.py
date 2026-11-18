"""Compare three category-aware score adjustments on two chronological pseudo-Public slices."""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from src.common.cv import pseudo_split
from src.common.metric import per_feed_ndcg
from src.common.preparation import write_json
from src.common.scores import feed_zscore


def probe(run, output):
    report = json.loads((run / 'run.json').read_text())
    if not report['complete'] or report['identity']['folds'] != 4:
        raise ValueError('a complete four-fold OOF run is required')
    flat = pl.scan_parquet(str(run / 'flat/val/*.parquet'))
    times = flat.select('feed_id', 'timestamp').unique().collect()
    pub, _ = pseudo_split(times['feed_id'], times['timestamp'])
    times = times.filter(pl.col('feed_id').is_in(pub))
    early, late = pseudo_split(times['feed_id'], times['timestamp'], frac=.5)
    keys = ['feed_id', 'item_id', 'candidate_index', 'target']
    rows = flat.filter(pl.col('feed_id').is_in(pub)).select(*keys, 'category_id', 'f1',
        'h_cat_wshare', 'cat_share_feed').collect()
    scores = pl.scan_parquet(run / 'validation_predictions.parquet').filter(pl.col('feed_id').is_in(pub)).collect()
    frame = rows.join(scores, on=keys, how='inner', validate='1:1').sort('feed_id', 'candidate_index')
    if frame.height != rows.height or frame.height != scores.height:
        raise ValueError('OOF coverage or targets differ')
    group = ['feed_id', 'category_id']
    frame = frame.with_columns(
        ((pl.col('f1').rank('average').over(group) - .5) / pl.len().over(group)).alias('within_category_rank'),
        pl.col('f1').top_k(3).mean().over(group).alias('category_top3_mean'),
        ((pl.col('h_cat_wshare') + .01) / (pl.col('cat_share_feed') + .01)).log().alias('history_to_supply'))
    feed, target = frame['feed_id'].to_numpy(), frame['target'].to_numpy()
    raw = frame['score'].to_numpy()
    base = feed_zscore(feed, raw)
    def evaluate(score, mask):
        return per_feed_ndcg(feed[mask], target[mask], score[mask], cutoff=30)[1]
    early_mask, late_mask = np.isin(feed, early), np.isin(feed, late)
    first, second = evaluate(raw, early_mask), evaluate(raw, late_mask)
    public_mean = float((first.sum() + second.sum()) / len(pub))
    if not np.isclose(public_mean, report['validation']['pseudo_public']['ndcg@30'], rtol=0, atol=1e-10):
        raise ValueError('baseline pseudo-Public metric differs')
    grid = []
    selected_score, best = raw, float(first.mean())
    chosen = {'signal': 'baseline', 'weight': 0.}
    for name in ('within_category_rank', 'category_top3_mean', 'history_to_supply'):
        signal = feed_zscore(feed, frame[name].to_numpy())
        for weight in (.03, .1, .3):
            adjusted = base + weight * signal
            mean = float(evaluate(adjusted, early_mask).mean())
            grid.append({'signal': name, 'weight': weight, 'early_ndcg30': mean})
            if mean > best:
                best, selected_score, chosen = mean, adjusted, {'signal': name, 'weight': weight}
    delta = evaluate(selected_score, late_mask) - second
    rng = np.random.default_rng(42)
    bootstrap = [float(rng.choice(delta, size=len(delta), replace=True).mean()) for _ in range(1000)]
    # Explanatory controls at the fixed weight, outside the selection grid.
    whole = feed_zscore(feed, frame.select(pl.col('f1').rank('average').over('feed_id')).to_series().to_numpy())
    category = feed_zscore(feed, frame['within_category_rank'].to_numpy())
    controls = {}
    for label, signal in [('whole_feed_f1_rank', whole), ('category_minus_whole_rank', category - whole)]:
        controls[label] = {'weight': .1,
            'early_delta': float((evaluate(base + .1 * signal, early_mask) - first).mean()),
            'late_delta': float((evaluate(base + .1 * signal, late_mask) - second).mean())}
    result = {'run': run.name, 'selection': 'early half of pseudo-Public; baseline or largest mean NDCG@30',
              'early_feeds': len(early), 'late_feeds': len(late), 'early_baseline': float(first.mean()),
              'grid': grid, 'chosen': chosen, 'late_baseline': float(second.mean()),
              'late_adjusted': float((second + delta).mean()), 'late_delta': float(delta.mean()),
              'paired_feed_bootstrap_ci95': np.quantile(bootstrap, [.025, .975]).tolist(),
              'explanatory_controls_after_selection': controls, 'pseudo_private_evaluated': False}
    write_json(output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    probe(args.run, args.out)
