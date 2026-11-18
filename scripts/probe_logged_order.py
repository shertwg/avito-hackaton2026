"""Learn a train-only proxy for logged display order and test its incremental ranking signal."""
import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from src.common.cv import pseudo_split
from src.common.metric import per_feed_ndcg
from src.common.preparation import fingerprint, write_json
from src.common.scores import feed_zscore


KEY = ['feed_id', 'item_id', 'candidate_index']
FIELDS = [f'f{i}{suffix}' for i in range(1, 6) for suffix in ('', '_rank', '_z')] + [
    'cos_user', 'cos_hist_max_10', 'cos_hist_max_50', 'h_cat_wshare', 'h_cat_rank',
    'i2i_sum_10', 'i2i_sum_50', 'h_item_cnt', 'cat_share_feed', 'feed_size']


def train_part(run, prepared, name, identity):
    flat = run / 'flat/train' / name
    source = prepared / 'train/candidates' / name
    if fingerprint(source) != identity['inputs'][str(source.resolve())]:
        raise ValueError('source candidate file changed')
    # Sample whole feeds before computing relative position.
    sample = pl.col('feed_id').hash(seed=42) % 16 == 0
    rows = pl.scan_parquet(flat).filter(sample).select(*KEY, *FIELDS).collect()
    positions = pl.scan_parquet(source).filter(sample).select(*KEY, 'pos').collect()
    joined = rows.join(positions, on=KEY, validate='1:1')
    if joined.height != rows.height or joined.height != positions.height or joined['pos'].null_count():
        raise ValueError('incomplete position labels')
    return joined.with_columns((1 - (pl.col('pos').rank('average').over('feed_id') - .5)
                                / pl.len().over('feed_id')).alias('relative_position'))


def probe(run, prepared, output, model_path):
    report = json.loads((run / 'run.json').read_text())
    identity = report['identity']
    names = identity['train_sources']
    if not report['complete'] or len(names) != 8 or identity['folds'] != 4:
        raise ValueError('expected finished 8-shard, four-fold OOF')
    train = pl.concat([train_part(run, prepared, name, identity) for name in names[:-1]])
    held = train_part(run, prepared, names[-1], identity)
    if set(train['feed_id']).intersection(held['feed_id']):
        raise ValueError('proxy train/holdout overlap')
    params = dict(objective='regression', learning_rate=.07, num_leaves=31, min_data_in_leaf=100,
                  verbosity=-1, num_threads=1, seed=42, deterministic=True, force_col_wise=True)
    model = lgb.train(params, lgb.Dataset(train.select(FIELDS).to_numpy(), label=train['relative_position'].to_numpy(),
                      feature_name=FIELDS), num_boost_round=100)
    pred = model.predict(held.select(FIELDS).to_numpy(), num_threads=1)
    held = held.with_columns(pl.Series('proxy', pred)).with_columns([
        (pl.col(k).rank('average').over('feed_id')).alias(k) for k in ('proxy', 'relative_position', 'f1')])
    corr = held.group_by('feed_id').agg([pl.corr(k, 'relative_position').alias(k) for k in ('proxy', 'f1')])
    flat = pl.scan_parquet(str(run / 'flat/val/*.parquet'))
    times = flat.select('feed_id', 'timestamp').unique().collect()
    pub, _ = pseudo_split(times['feed_id'], times['timestamp'])
    times = times.filter(pl.col('feed_id').is_in(pub))
    early, late = pseudo_split(times['feed_id'], times['timestamp'], .5)
    rows = flat.filter(pl.col('feed_id').is_in(pub)).select(*KEY, 'target', *FIELDS).collect()
    scores = pl.scan_parquet(run / 'validation_predictions.parquet').filter(pl.col('feed_id').is_in(pub)).collect()
    val = rows.join(scores, on=[*KEY, 'target'], validate='1:1').sort('feed_id', 'candidate_index')
    if rows.height != val.height or scores.height != val.height or set(val['feed_id']).intersection(train['feed_id']):
        raise ValueError('val alignment or train separation failed')
    feed, target = val['feed_id'].to_numpy(), val['target'].to_numpy()
    base = feed_zscore(feed, val['score'])
    proxy = feed_zscore(feed, model.predict(val.select(FIELDS).to_numpy(), num_threads=1))
    def evaluate(ids, weight):
        mask = np.isin(feed, ids)
        return per_feed_ndcg(feed[mask], target[mask], (base + weight * proxy)[mask], cutoff=30)[1]
    grid = [(weight, float(evaluate(early, weight).mean())) for weight in (0., .05, .15, .3, .6)]
    weight, _ = max(grid, key=lambda pair: pair[1])
    baseline = evaluate(late, 0)
    if abs((grid[0][1] * len(early) + baseline.sum()) / len(pub) -
           report['validation']['pseudo_public']['ndcg@30']) > 1e-10:
        raise ValueError('baseline metric mismatch')
    delta = evaluate(late, weight) - baseline
    rng = np.random.default_rng(42)
    ci = np.quantile([rng.choice(delta, len(delta)).mean() for _ in range(1000)], [.025, .975])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    result = dict(run=run.name, fields=FIELDS, proxy_params=params, rounds=100, train_rows=train.height,
                  train_feeds=train['feed_id'].n_unique(), train_sources=names[:-1], held_source=names[-1],
                  held_rows=held.height, held_feeds=corr.height,
                  held_mean_within_feed_spearman={k: float(corr[k].filter(corr[k].is_finite()).mean())
                                                   for k in ('proxy','f1')},
                  early_feeds=len(early), late_feeds=len(late), public_early_grid=grid, selected_weight=weight,
                  late_baseline=float(baseline.mean()), late_adjusted=float((baseline + delta).mean()),
                  late_delta=float(delta.mean()), paired_feed_ci95=ci.tolist(), pseudo_private_evaluated=False)
    write_json(output, result)
    print(json.dumps(result, indent=2))


def confirm(run, public_report, model_path, output):
    config = json.loads(public_report.read_text())
    if config['run'] != run.name or config['fields'] != FIELDS:
        raise ValueError('selection report differs from inference run')
    model = lgb.Booster(model_file=str(model_path))
    scores = pl.read_parquet(run / 'validation_predictions.parquet')
    chunks = []
    for shard in sorted((run / 'flat/val').glob('*.parquet')):
        part = pl.read_parquet(shard, columns=[*KEY, 'timestamp', 'target', *FIELDS])
        pred = model.predict(part.select(FIELDS).to_numpy(), num_threads=1)
        chunks.append(part.select(*KEY, 'timestamp', 'target').with_columns(pl.Series('proxy', pred)))
    data = pl.concat(chunks).join(scores, on=[*KEY, 'target'], validate='1:1').sort('feed_id', 'candidate_index')
    if data.height != scores.height or data.height != sum(chunk.height for chunk in chunks):
        raise ValueError('candidate alignment failed')
    pub, priv = pseudo_split(data['feed_id'], data['timestamp'])
    weight = config['selected_weight']
    def metrics(ids):
        part = data.filter(pl.col('feed_id').is_in(ids))
        f, t = part['feed_id'].to_numpy(), part['target'].to_numpy()
        mixed = feed_zscore(f, part['score']) + weight * feed_zscore(f, part['proxy'])
        return [per_feed_ndcg(f, t, s, cutoff=30)[1] for s in (part['score'].to_numpy(), mixed)]
    subset = data.filter(pl.col('feed_id').is_in(pub)).select('feed_id', 'timestamp').unique()
    _, late = pseudo_split(subset['feed_id'], subset['timestamp'], .5)
    if abs(metrics(late)[1].mean() - config['late_adjusted']) > 1e-10:
        raise ValueError('saved model failed to reproduce public result')
    base, mixed = metrics(priv)
    expected = json.loads((run / 'run.json').read_text())['validation']['pseudo_private']['ndcg@30']
    if abs(base.mean() - expected) > 1e-10:
        raise ValueError('private baseline differs')
    delta = mixed - base
    rng = np.random.default_rng(42)
    result = dict(selection_report_sha256=hashlib.sha256(public_report.read_bytes()).hexdigest(),
                  model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                  fixed_weight=weight, feeds=len(priv), private_baseline=float(base.mean()),
                  private_adjusted=float(mixed.mean()), delta=float(delta.mean()),
                  paired_feed_ci95=np.quantile([rng.choice(delta, len(delta)).mean() for _ in range(1000)],
                                               [.025, .975]).tolist())
    write_json(output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, default=Path('data/prepared'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model-out', type=Path, required=True)
    parser.add_argument('--confirm-from', type=Path, help='evaluate saved model at previously selected weight')
    args = parser.parse_args()
    if args.confirm_from:
        confirm(args.run, args.confirm_from, args.model_out, args.out)
    else:
        probe(args.run, args.prepared, args.out, args.model_out)
