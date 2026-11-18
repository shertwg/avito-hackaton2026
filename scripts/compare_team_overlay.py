"""Measure the incremental NDCG@30 of one model added to a frozen team blend."""
import argparse
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import polars as pl

from scripts.validate_submission import validate
from src.common.cv import pseudo_split
from src.common.data import load_labels
from src.common.metric import per_feed_ndcg
from src.common.preparation import write_json
from src.common.scores import feed_zscore


KEY = ['feed_id', 'item_id']
WEIGHTS = (0., .025, .05, .1, .15, .2, .3, .4, .5, .75, 1.)


def aligned(reference, path):
    values = pl.read_parquet(path)
    if not all(values.schema[k].is_integer() for k in KEY):
        raise ValueError(f'noninteger candidate keys: {path}')
    values = values.with_columns(pl.col(KEY).cast(pl.Int32)).sort(KEY)
    if not values.select(KEY).equals(reference.select(KEY)):
        raise ValueError(f'candidate keys differ from reference: {path}')
    if 'target' in values and 'target' in reference and not values['target'].equals(reference['target']):
        raise ValueError(f'validation labels differ: {path}')
    if values['score'].dtype not in (pl.Float32, pl.Float64) or values['score'].null_count() or not values['score'].is_finite().all():
        raise ValueError(f'invalid scores: {path}')
    return values['score'].to_numpy()


def mix(raw, base_z, new_z, weight):
    return raw if weight == 0 else ((1 - weight) * base_z + weight * new_z).astype(np.float32)


def compare(raw_root, base, candidate, out):
    if out.exists():
        raise FileExistsError(f'choose a fresh output directory: {out}')
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = [p / f'{split}_pred.parquet' for p in (base, candidate) for split in ('val', 'test')]
    before = _hash_files(paths)
    results = {'base': base.name, 'candidate': candidate.name, 'input_sha256': before}
    with TemporaryDirectory(dir=out.parent) as tmp:
        stage = Path(tmp) / 'result'
        stage.mkdir()
        for split in ('val', 'test'):
            template = raw_root / 'sample_submission.parquet'
            reference = (load_labels(raw_root, split) if split == 'val' else pl.read_parquet(template, columns=KEY))
            reference = reference.with_columns(pl.col(KEY).cast(pl.Int32)).sort(KEY)
            if reference.select(KEY).is_duplicated().any() or reference.null_count().to_numpy().any():
                raise ValueError('duplicate keys or nulls in reference')
            feed = reference['feed_id'].to_numpy()
            raw, other = [aligned(reference, root / f'{split}_pred.parquet') for root in (base, candidate)]
            z, extra = feed_zscore(feed, raw), feed_zscore(feed, other)
            if split == 'val':
                pub, priv = pseudo_split(feed, reference['timestamp'])
                if not len(pub) or not len(priv):
                    raise ValueError('both temporal slices must contain feeds')
                target = reference['target'].to_numpy()
                def metric(mask, w):
                    s = mix(raw[mask], z[mask], extra[mask], w)
                    return per_feed_ndcg(feed[mask], target[mask], s, cutoff=30)[1]
                early, late = np.isin(feed, pub), np.isin(feed, priv)
                grid = [float(metric(early, w).mean()) for w in WEIGHTS]
                weight = WEIGHTS[int(np.argmax(grid))]
                original, updated = metric(late, 0), metric(late, weight)
                delta = updated - original
                rng = np.random.default_rng(42)
                draws = [rng.choice(delta, len(delta)).mean() for _ in range(1000)]
                results.update(public_grid=list(zip(WEIGHTS, grid)), candidate_weight=weight,
                               public_feeds=len(pub), private_feeds=len(priv), private_base=float(original.mean()),
                               private_blend=float(updated.mean()), private_delta=float(delta.mean()),
                               paired_feed_bootstrap_ci95=np.quantile(draws, [.025, .975]).tolist())
            frame = reference.select(KEY).with_columns(pl.Series('score', mix(raw, z, extra, weight)))
            if split == 'val':
                frame = frame.with_columns(reference['target'])
            else:
                frame = frame.with_columns(pl.col('score').cast(pl.Float32))
            frame.write_parquet(stage / f'{split}_pred.parquet')
            results[f'{split}_rows'] = frame.height
        errors = validate(str(template), str(stage / 'test_pred.parquet'))
        if errors or before != _hash_files(paths):
            raise ValueError(errors or 'input predictions changed during comparison')
        results['submission_validator'] = 'passed'
        write_json(stage / 'comparison.json', results)
        stage.rename(out)
    return results


def _hash_files(paths):
    result = {}
    for path in paths:
        with path.open('rb') as stream:
            result[str(path)] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('competition'))
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.data, args.base, args.candidate, args.out)
    print('Candidate weight:', report['candidate_weight'], 'Private delta:', report['private_delta'])
