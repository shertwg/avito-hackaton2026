"""Sequential time-block CatBoost OOF with disk quantization and fold checkpoints."""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

from catboost import CatBoostRanker, Pool
import duckdb
import numpy as np
import polars as pl

from scripts.run_baseline_check import git_snapshot, pool
from scripts.validate_submission import validate
from src.common.cv import pseudo_split, time_folds
from src.common.feature_cache import cache_lock
from src.common.metric import all_variants
from src.common.preparation import fingerprint, write_json, write_parquet_atomic
from src.common.resources import memory_snapshot
from src.common.scores import feed_zscore
from src.common.similarity import attach_features
from src.gbdt.prepared import BASE, select_train_parts


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def load_part(prepared, entry, split, roots):
    schema = pl.read_parquet_schema(prepared / entry['candidate_path'])
    columns = ['feed_id', 'candidate_index', 'item_id', 'timestamp', *BASE]
    if 'target' in schema:
        columns.append('target')
    frame = pl.read_parquet(prepared / entry['candidate_path'], columns=columns)
    for root in roots:
        frame, _ = attach_features(frame, root, split, [entry['source']])
    return frame.sort('feed_id', 'candidate_index')


def export_training(flat_paths, assignments, held_fold, features, target, *, memory_mb, threads):
    connection = duckdb.connect(config={'memory_limit': f'{memory_mb}MB', 'threads': str(threads),
                                      'temp_directory': str(target.parent / '_spill')})
    try:
        connection.from_parquet([str(p) for p in flat_paths]).create_view('candidates')
        connection.register('assignments', assignments.to_arrow())
        names = ','.join('"' + name.replace('"', '""') + '"' for name in features)
        relation = connection.sql(
            f'SELECT target,c.feed_id,{names} FROM candidates c LEFT JOIN assignments a USING(feed_id) '
            f'WHERE a.fold IS NULL OR a.fold!={int(held_fold)} ORDER BY c.feed_id,c.candidate_index')
        relation.write_csv(str(target), sep='\t', header=False)
    finally:
        connection.close()


def run(prepared, raw, output, roots, *, train_parts=8, folds=4, iterations=92, depth=6,
        learning_rate=.1, seed=42, threads=2, memory_mb=512, resume=False,
        reference=Path('competition/baseline/calculate_metrics.py')):
    prepared, raw, output = [Path(p).resolve() for p in (prepared, raw, output)]
    roots = [Path(p).resolve() for p in roots]
    if any(output == path or output in path.parents or path in output.parents for path in (prepared, raw, *roots)):
        raise ValueError('OOF output must be separate from data and feature caches')
    if (folds < 2 or iterations < 1 or threads < 1 or memory_mb < 128 or depth < 1
            or not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError('invalid CV resource or model parameters')
    source = json.loads((prepared / 'manifest.json').read_text(encoding='utf-8'))
    if not source.get('complete'):
        raise ValueError('prepared data incomplete')
    selected = select_train_parts(prepared, train_parts)
    entries = {'train': selected, 'val': source['splits']['val']['parts'], 'test': source['splits']['test']['parts']}
    features, groups, inputs = list(BASE), [], {}
    for root in roots:
        manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
        if not manifest.get('complete') or set(features) & set(manifest['features']):
            raise ValueError('incomplete or duplicate feature group')
        if 'excluded_train_sources' in manifest and not {p['source'] for p in selected} <= set(manifest['excluded_train_sources']):
            raise ValueError('donor shards overlap requested training cohort')
        features.extend(manifest['features'])
        groups.append({'root': str(root), 'manifest_sha256': digest(root / 'manifest.json')})
        inputs[str(root / 'manifest.json')] = fingerprint(root / 'manifest.json')
        for split, parts in entries.items():
            for part in parts:
                cached = manifest['splits'][split]['parts'][part['source']]
                path = root / cached['path']
                if fingerprint(path) != cached['fingerprint']:
                    raise ValueError('feature file changed since its manifest')
                inputs[str(path)] = fingerprint(path)
    for parts in entries.values():
        for entry in parts:
            for field in ('candidate_path', 'feed_path'):
                path = prepared / entry[field]
                inputs[str(path)] = fingerprint(path)
    inputs[str(prepared / 'manifest.json')] = fingerprint(prepared / 'manifest.json')
    identity = {'implementation_sha256': digest(__file__), 'prepared_manifest_sha256': digest(prepared / 'manifest.json'),
                'groups': groups, 'features': features, 'inputs': inputs, 'folds': folds,
                'train_sources': [p['source'] for p in selected], 'iterations': iterations,
                'depth': depth, 'learning_rate': learning_rate, 'seed': seed, 'threads': threads,
                'reference_sha256': digest(reference), 'sample_submission': fingerprint(raw / 'sample_submission.parquet'),
                'code_sha256': {name: digest(name) for name in ('scripts/run_baseline_check.py', 'src/common/cv.py',
                    'src/common/metric.py', 'src/common/scores.py', 'src/common/similarity.py', 'src/gbdt/prepared.py')}}
    metadata = {}
    for split in ('train', 'val'):
        metadata[split] = pl.concat([pl.read_parquet(prepared / p['candidate_path'],
            columns=['feed_id', 'candidate_index', 'item_id', 'timestamp', 'target']) for p in entries[split]]).sort('feed_id', 'candidate_index')
    train_ids = set(metadata['train']['feed_id'].unique().to_list())
    val = metadata['val']
    if train_ids & set(val['feed_id'].unique().to_list()):
        raise ValueError('train and validation feed identifiers overlap')
    unique, fold = time_folds(val['feed_id'].to_numpy(), val['timestamp'].to_numpy(), folds)
    if set(fold) != set(range(folds)):
        raise ValueError('not enough feeds for nonempty folds')
    assignments = pl.DataFrame({'feed_id': unique, 'fold': fold})
    del metadata['train']
    output.mkdir(parents=True, exist_ok=True)
    with cache_lock(output):
        report_path = output / 'run.json'
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding='utf-8'))
            if not resume or report['identity'] != identity:
                raise ValueError('existing OOF run requires matching identity and --resume')
            if report.get('complete'):
                for name, stamp in report['outputs'].items():
                    if fingerprint(output / name) != stamp:
                        raise ValueError('completed OOF artifact changed')
                return report
        else:
            if any(p.name != '_build.lock' for p in output.iterdir()):
                raise ValueError('nonempty OOF directory has no report')
            report = {'identity': identity, 'git': git_snapshot(), 'complete': False, 'flat': {}, 'folds': {},
                      'started_unix': time.time(), 'purpose': 'real time-block OOF; fixed joint-008 parameters'}

        def checkpoint(stage):
            report.update(stage=stage, resources=memory_snapshot(), elapsed_seconds=round(time.time() - report['started_unix'], 2))
            write_json(report_path, report)
            print(f'OOF {stage}, {report["elapsed_seconds"]:.0f}s', flush=True)

        checkpoint('preparing_flat_cache')
        write_parquet_atomic(assignments, output / 'fold_assignments.parquet')
        flat_paths = []
        for split in ('train', 'val'):
            for entry in entries[split]:
                relative = f'flat/{split}/{entry["source"]}'
                path = output / relative
                stamp = report['flat'].get(relative)
                if not path.exists() or fingerprint(path) != stamp:
                    frame = load_part(prepared, entry, split, roots)
                    write_parquet_atomic(frame, path)
                    report['flat'][relative] = fingerprint(path)
                    del frame
                    checkpoint(f'flat_{split}_{len(flat_paths) + 1}')
                flat_paths.append(path)
        cd = output / 'columns.cd'
        cd.write_text('0\tLabel\n1\tGroupId\n' + ''.join(
            f'{i+2}\t{"Categ" if name=="category_id" else "Num"}\t{name}\n' for i, name in enumerate(features)), encoding='utf-8')
        for number in range(folds):
            tag = str(number)
            directory = output / f'fold-{number}'
            directory.mkdir(exist_ok=True)
            saved = report['folds'].setdefault(tag, {})
            if saved.get('complete'):
                if all(fingerprint(directory / name) == stamp for name, stamp in saved['outputs'].items()):
                    continue
                raise ValueError('completed fold artifacts changed')
            quantized = directory / 'train.qbin'
            if not quantized.exists() or saved.get('quantized') != fingerprint(quantized):
                tsv = directory / 'train.tsv'
                checkpoint(f'fold_{number}_exporting')
                export_training(flat_paths, assignments, number, features, tsv, memory_mb=memory_mb, threads=threads)
                checkpoint(f'fold_{number}_quantizing')
                data = Pool(str(tsv), column_description=str(cd), thread_count=threads)
                data.quantize(used_ram_limit=f'{memory_mb}mb', border_count=254, random_seed=seed)
                data.save(str(quantized) + '.tmp')
                del data
                gc.collect()
                Path(str(quantized) + '.tmp').replace(quantized)
                saved['quantized'] = fingerprint(quantized)
                tsv.unlink()
                checkpoint(f'fold_{number}_quantized')
            data = Pool('quantized://' + str(quantized), thread_count=threads)
            held_ids = unique[fold == number]
            expected_rows = sum(p['candidates'] for p in selected) + val.filter(~pl.col('feed_id').is_in(held_ids)).height
            if data.num_row() != expected_rows or data.get_feature_names() != features:
                raise ValueError('quantized training rows or feature names differ')
            model = CatBoostRanker(loss_function='YetiRank', iterations=iterations, depth=depth,
                learning_rate=learning_rate, random_seed=seed, thread_count=threads, used_ram_limit='1024mb',
                eval_metric='NDCG:top=30;type=Base', allow_writing_files=True,
                train_dir=str(directory / 'catboost'), verbose=20)
            model_path = directory / 'model.cbm'
            if not model_path.exists() or saved.get('trained_model') != fingerprint(model_path):
                checkpoint(f'fold_{number}_fit')
                model.fit(data, save_snapshot=True, snapshot_file=str(directory / 'snapshot.bin'), snapshot_interval=300)
                model.save_model(str(directory / 'model.cbm.tmp'))
                Path(directory / 'model.cbm.tmp').replace(model_path)
                saved['trained_model'] = fingerprint(model_path)
                checkpoint(f'fold_{number}_trained')
            del model, data
            gc.collect()
            model = CatBoostRanker()
            model.load_model(str(directory / 'model.cbm'))
            checkpoint(f'fold_{number}_predicting')
            predictions = []
            for entry in entries['val']:
                frame = pl.read_parquet(output / f'flat/val/{entry["source"]}').filter(pl.col('feed_id').is_in(held_ids))
                if frame.height:
                    scores = model.predict(pool(frame, labeled=False, features=features), thread_count=threads)
                    predictions.append(frame.select('feed_id', 'candidate_index', 'item_id', 'target').with_columns(pl.Series('score', scores)))
            write_parquet_atomic(pl.concat(predictions).sort('feed_id', 'candidate_index'), directory / 'val.parquet')
            predictions = []
            for entry in entries['test']:
                frame = load_part(prepared, entry, 'test', roots)
                scores = model.predict(pool(frame, labeled=False, features=features), thread_count=threads)
                scores = feed_zscore(frame['feed_id'].to_numpy(), scores)
                predictions.append(frame.select('feed_id', 'candidate_index', 'item_id').with_columns(pl.Series('score', scores)))
            write_parquet_atomic(pl.concat(predictions).sort('feed_id', 'candidate_index'), directory / 'test.parquet')
            saved.update(complete=True, train_rows=expected_rows, held_feeds=len(held_ids),
                         outputs={name: fingerprint(directory / name) for name in ('model.cbm', 'val.parquet', 'test.parquet')})
            del model, predictions, frame
            gc.collect()
            checkpoint(f'fold_{number}_complete')
        oof = pl.concat([pl.read_parquet(output / f'fold-{i}/val.parquet') for i in range(folds)]).sort('feed_id', 'candidate_index')
        if not oof.select('feed_id', 'candidate_index', 'item_id', 'target').equals(val.select('feed_id', 'candidate_index', 'item_id', 'target')):
            raise ValueError('OOF prediction keys, labels or coverage changed')
        oof = oof.with_columns(pl.col('score').cast(pl.Float32))
        if not oof['score'].is_finite().all():
            raise ValueError('nonfinite OOF scores')
        write_parquet_atomic(oof, output / 'validation_predictions.parquet')
        write_parquet_atomic(oof.select('feed_id', 'item_id', 'score', 'target'), output / 'val_pred.parquet')
        test = pl.read_parquet(output / 'fold-0/test.parquet')
        total = test['score'].to_numpy().copy()
        for i in range(1, folds):
            other = pl.read_parquet(output / f'fold-{i}/test.parquet')
            if not other.select('feed_id', 'candidate_index', 'item_id').equals(test.select('feed_id', 'candidate_index', 'item_id')):
                raise ValueError('test fold occurrence keys changed')
            total += other['score'].to_numpy()
        test = test.select('feed_id', 'item_id').with_columns(pl.Series('score', (total / folds).astype(np.float32)))
        write_parquet_atomic(test, output / 'test_pred.parquet')
        errors = validate(str(raw / 'sample_submission.parquet'), str(output / 'test_pred.parquet'))
        if errors:
            raise ValueError(errors)
        spec = importlib.util.spec_from_file_location('official_evaluator', reference)
        evaluator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluator)
        public, private = pseudo_split(val['feed_id'].to_numpy(), val['timestamp'].to_numpy())
        report['validation'] = {}
        for name, ids in (('all', unique), ('pseudo_public', public), ('pseudo_private', private)):
            sample = oof.filter(pl.col('feed_id').is_in(ids))
            scores = all_variants(sample['feed_id'], sample['target'], sample['score'])
            official = evaluator.metrics(sample)
            metric_keys = [('ndcg_full', 'ndcg'), ('ndcg@10', 'ndcg_at_10'), ('ndcg@30', 'ndcg_at_30')]
            if any(abs(scores[ours] - official[theirs]) > 1e-10 for ours, theirs in metric_keys):
                raise ValueError('official metric disagreement')
            report['validation'][name] = {'feeds': len(ids), **scores, 'reference_check': 'passed'}
        if any(fingerprint(Path(path)) != stamp for path, stamp in inputs.items()):
            raise ValueError('source inputs changed during OOF')
        report.update(complete=True, submission_validator='passed',
                      outputs={name: fingerprint(output / name) for name in ('val_pred.parquet', 'test_pred.parquet', 'validation_predictions.parquet')},
                      prediction_sha256={name: digest(output / name) for name in ('val_pred.parquet', 'test_pred.parquet')})
        checkpoint('complete')
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, default=Path('data/prepared'))
    parser.add_argument('--raw', type=Path, default=Path('competition'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--feature-root', type=Path, action='append', required=True)
    parser.add_argument('--train-parts', type=int, default=8)
    parser.add_argument('--folds', type=int, default=4)
    parser.add_argument('--iterations', type=int, default=92)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--learning-rate', type=float, default=.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--memory-mb', type=int, default=512)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.prepared, args.raw, args.out, args.feature_root, train_parts=args.train_parts,
        folds=args.folds, iterations=args.iterations, depth=args.depth, learning_rate=args.learning_rate,
        seed=args.seed, threads=args.threads, memory_mb=args.memory_mb, resume=args.resume)


if __name__ == '__main__':
    main()
