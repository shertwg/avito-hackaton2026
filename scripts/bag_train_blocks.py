"""Average audited four-fold predictions from all disjoint training blocks."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import polars as pl

from scripts.compare_team_overlay import KEY, aligned
from scripts.run_joint_oof import digest
from scripts.validate_submission import validate
from src.common.cv import pseudo_split, time_folds
from src.common.data import load_labels
from src.common.metric import per_feed_ndcg
from src.common.preparation import write_json
from src.common.scores import feed_zscore


def bag(plan, out):
    if out.exists():
        raise FileExistsError(out)
    layout = json.loads(plan.read_bytes())
    identity = layout['identity']
    runs = Path(identity['oof_out'])
    raw = Path(identity['raw'])
    queue = json.loads((runs / 'blocks_run.json').read_bytes())
    if (not layout['complete'] or not queue['complete'] or queue['identity']['plan_sha256'] != digest(plan)
            or set(queue['blocks']) != set(layout['blocks']) or len(queue['blocks']) < 2):
        raise ValueError('all planned blocks must have completed OOF predictions')
    val = load_labels(raw, 'val').with_columns(pl.col(KEY).cast(pl.Int32)).sort(KEY)
    ids, folds = time_folds(val['feed_id'].to_numpy(), val['timestamp'].to_numpy(), 4)
    expected_folds = pl.DataFrame({'feed_id': ids, 'fold': folds})
    sources = set()
    hashes = {}
    for tag, block in layout['blocks'].items():
        saved = json.loads((runs / tag / 'run.json').read_bytes())
        config = saved['identity']
        cohort = block['cohort']['sources']
        if (not saved['complete'] or saved['submission_validator'] != 'passed' or config['folds'] != 4
                or config['prepared_manifest_sha256'] != identity['prepared_manifest_sha256']
                or config['train_sources'] != cohort or sources.intersection(cohort)
                or len(set(cohort)) != len(cohort)
                or not pl.read_parquet(runs / tag / 'fold_assignments.parquet').sort('feed_id').equals(expected_folds)
                or saved['prediction_sha256'] != queue['blocks'][tag]['prediction_sha256']):
            raise ValueError(f'block cohort, folds or predictions differ: {tag}')
        sources.update(cohort)
        hashes[tag] = saved['prediction_sha256']
    template = raw / 'sample_submission.parquet'
    out.parent.mkdir(parents=True, exist_ok=True)
    report = dict(plan_sha256=digest(plan), components=hashes, weight_per_block=1/len(hashes))
    with TemporaryDirectory(dir=out.parent) as tmp:
        stage = Path(tmp) / 'result'
        stage.mkdir()
        for split in ('val', 'test'):
            reference = val if split == 'val' else pl.read_parquet(template, columns=KEY).sort(KEY)
            if reference.select(KEY).is_duplicated().any() or reference.null_count().to_numpy().any():
                raise ValueError('invalid reference keys or labels')
            feed = reference['feed_id'].to_numpy()
            summed = np.zeros(len(feed), dtype=np.float64)
            filename = f'{split}_pred.parquet'
            for tag, expected in hashes.items():
                path = runs / tag / filename
                if digest(path) != expected[filename]:
                    raise ValueError(f'prediction hash differs: {path}')
                summed += feed_zscore(feed, aligned(reference, path))
            score = (summed / len(hashes)).astype(np.float32)
            frame = reference.select(KEY).with_columns(pl.Series('score', score))
            if split == 'val':
                frame = frame.with_columns(val['target'])
                public, private = pseudo_split(feed, val['timestamp'])
                ids, ndcg = per_feed_ndcg(feed, val['target'].to_numpy(), score, cutoff=30)
                report['ndcg30'] = {key: float(ndcg[np.isin(ids, subset)].mean())
                                    for key, subset in [('pseudo_public', public), ('pseudo_private', private)]}
            frame.write_parquet(stage / filename)
            report[f'{split}_rows'] = frame.height
        errors = validate(str(template), str(stage / 'test_pred.parquet'))
        if errors:
            raise ValueError(errors)
        report.update(submission_validator='passed', prediction_sha256={
            name: digest(stage / name) for name in ('val_pred.parquet', 'test_pred.parquet')})
        write_json(stage / 'report.json', report)
        stage.rename(out)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(bag(args.plan, args.out)['ndcg30'], indent=2))
