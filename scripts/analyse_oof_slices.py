"""Describe validated OOF errors and unlabeled test coverage by fixed feed slices."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

from src.common.cv import pseudo_split, time_folds
from src.common.metric import per_feed_ndcg
from src.common.preparation import fingerprint, write_json


def summarize(frame):
    result = {'feeds': frame.height, 'candidates': int(frame['candidates'].sum()),
              'mean_candidates': float(frame['candidates'].mean()),
              'empty_history_fraction': float((frame['history_length'] == 0).mean()),
              'mean_candidate_embedding_fraction': float(frame['embedding_fraction'].mean())}
    if 'ndcg30' in frame:
        positives = frame.filter(pl.col('positives') > 0)
        result.update(ndcg30=float(frame['ndcg30'].mean()),
                      no_positive_feeds=int((frame['positives'] == 0).sum()),
                      ndcg30_positive_feeds=float(positives['ndcg30'].mean()) if positives.height else None)
    return result


def slices(frame):
    partitions = {
        'history_length': [
            ('0', pl.col('history_length') == 0),
            ('1..10', pl.col('history_length').is_between(1, 10)),
            ('11..50', pl.col('history_length').is_between(11, 50)),
            ('51+', pl.col('history_length') > 50)],
        'candidates': [
            ('1..29', pl.col('candidates') < 30),
            ('30..59', pl.col('candidates').is_between(30, 59)),
            ('60..99', pl.col('candidates').is_between(60, 99)),
            ('100..199', pl.col('candidates').is_between(100, 199)),
            ('200+', pl.col('candidates') >= 200)],
        'embedding_fraction': [
            ('0', pl.col('embedding_fraction') == 0),
            ('(0,.5)', (pl.col('embedding_fraction') > 0) & (pl.col('embedding_fraction') < .5)),
            ('[.5,1)', (pl.col('embedding_fraction') >= .5) & (pl.col('embedding_fraction') < 1)),
            ('1', pl.col('embedding_fraction') == 1)]}
    result = {}
    for name, definitions in partitions.items():
        values = {}
        for label, condition in definitions:
            part = frame.filter(condition)
            if part.height:
                values[label] = summarize(part)
        if sum(part['feeds'] for part in values.values()) != frame.height:
            raise ValueError('incomplete or overlapping diagnostic partition')
        result[name] = values
    return result


def analyse(prepared, cosine, run, output):
    prepared, cosine, run, output = map(lambda path: Path(path).resolve(), (prepared, cosine, run, output))
    source = json.loads((prepared / 'manifest.json').read_text())
    cache = json.loads((cosine / 'manifest.json').read_text())
    report = json.loads((run / 'run.json').read_text())
    if not source.get('complete') or not cache.get('complete') or not report.get('complete'):
        raise ValueError('complete prepared data, cosine cache and OOF run required')
    if len(report['folds']) != 4:
        raise ValueError('four-fold OOF run required')
    if hashlib.sha256((prepared / 'manifest.json').read_bytes()).hexdigest() != report['identity']['prepared_manifest_sha256']:
        raise ValueError('prepared source differs from OOF input')
    cosine_sha = hashlib.sha256((cosine / 'manifest.json').read_bytes()).hexdigest()
    if not any(group['root'] == str(cosine) and group['manifest_sha256'] == cosine_sha for group in report['identity']['groups']):
        raise ValueError('cosine cache differs from OOF input')
    if fingerprint(run / 'validation_predictions.parquet') != report['outputs']['validation_predictions.parquet']:
        raise ValueError('OOF predictions changed')
    before = {str(path): fingerprint(path) for path in (
        prepared / 'manifest.json', cosine / 'manifest.json', run / 'run.json',
        run / 'validation_predictions.parquet', run / 'fold_assignments.parquet')}
    predictions = pl.read_parquet(run / 'validation_predictions.parquet').rename({'target': 'prediction_target'})
    feed_frames = {}
    for split in ('val', 'test'):
        candidates, features = [], []
        for entry in source['splits'][split]['parts']:
            path = prepared / entry['candidate_path']
            before[str(path)] = fingerprint(path)
            if before[str(path)] != report['identity']['inputs'].get(str(path)):
                raise ValueError('prepared candidate file differs from OOF input')
            columns = ['feed_id', 'candidate_index', 'item_id', 'timestamp']
            if split == 'val':
                columns.append('target')
            candidates.append(pl.read_parquet(path, columns=columns))
            part = cache['splits'][split]['parts'][entry['source']]
            path = cosine / part['path']
            if fingerprint(path) != part['fingerprint']:
                raise ValueError('cosine feature file changed')
            before[str(path)] = part['fingerprint']
            features.append(pl.read_parquet(path, columns=[
                'feed_id', 'candidate_index', 'item_id', 'has_item_embedding', 'hist_len']))
        frame = pl.concat(candidates).join(pl.concat(features), on=['feed_id', 'candidate_index', 'item_id'],
                                          how='left', validate='1:1')
        if frame.null_count().to_numpy().any():
            raise ValueError('missing diagnostic candidate features')
        aggregates = [pl.len().alias('candidates'), pl.col('timestamp').first(),
                      pl.col('hist_len').first().alias('history_length'),
                      pl.col('has_item_embedding').mean().alias('embedding_fraction')]
        if split == 'val':
            frame = frame.join(predictions, on=['feed_id', 'candidate_index', 'item_id'], how='left', validate='1:1')
            if frame.height != predictions.height or frame.null_count().to_numpy().any():
                raise ValueError('OOF prediction coverage differs')
            if not frame['target'].equals(frame['prediction_target']) or not np.isfinite(frame['score'].to_numpy()).all():
                raise ValueError('OOF targets or scores differ')
            ids, values = per_feed_ndcg(frame['feed_id'].to_numpy(), frame['target'].to_numpy(), frame['score'].to_numpy(), cutoff=30)
            aggregates.append(pl.col('target').sum().alias('positives'))
        feeds = frame.group_by('feed_id').agg(aggregates)
        if split == 'val':
            feeds = feeds.join(pl.DataFrame({'feed_id': ids, 'ndcg30': values}), on='feed_id', validate='1:1')
            public, private = pseudo_split(feeds['feed_id'].to_numpy(), feeds['timestamp'].to_numpy())
            feeds = feeds.with_columns(pl.when(pl.col('feed_id').is_in(public)).then(pl.lit('pseudo_public'))
                                      .otherwise(pl.lit('pseudo_private')).alias('leaderboard_split'))
            for name, selected in (('all', feeds), ('pseudo_public', feeds.filter(pl.col('feed_id').is_in(public))),
                                   ('pseudo_private', feeds.filter(pl.col('feed_id').is_in(private)))):
                if abs(float(selected['ndcg30'].mean()) - report['validation'][name]['ndcg@30']) > 1e-12:
                    raise ValueError('OOF aggregate metric disagreement')
            assignments = pl.read_parquet(run / 'fold_assignments.parquet')
            ids, folds = time_folds(feeds['feed_id'].to_numpy(), feeds['timestamp'].to_numpy(), 4)
            expected = pl.DataFrame({'feed_id': ids, 'fold': folds})
            if not assignments.sort('feed_id').equals(expected):
                raise ValueError('OOF time-block assignments differ')
            feeds = feeds.join(assignments, on='feed_id', validate='1:1')
            if feeds.null_count().to_numpy().any():
                raise ValueError('OOF fold assignment coverage differs')
        feed_frames[split] = feeds
        del frame
    val, test = feed_frames['val'], feed_frames['test']
    subsets = {'all_val': val, 'pseudo_public': val.filter(pl.col('leaderboard_split') == 'pseudo_public'),
               'pseudo_private': val.filter(pl.col('leaderboard_split') == 'pseudo_private'), 'test': test}
    result = {'complete': True, 'scope': 'descriptive fixed feed partitions; no feature or model parameter selection',
              'inputs': before, 'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'subsets': {name: {'summary': summarize(frame), 'slices': slices(frame)} for name, frame in subsets.items()},
              'folds': {str(fold): summarize(val.filter(pl.col('fold') == fold)) for fold in range(4)}}
    if any(fingerprint(Path(path)) != stamp for path, stamp in before.items()):
        raise ValueError('diagnostic inputs changed')
    write_json(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, default=Path('data/prepared'))
    parser.add_argument('--cosine', type=Path, default=Path('data/features/cosine'))
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    arguments = parser.parse_args()
    result = analyse(arguments.prepared, arguments.cosine, arguments.run, arguments.out)
    print(json.dumps({name: value['summary'] for name, value in result['subsets'].items()}, indent=2))
