"""Prepare and audit disjoint joint/item2item train cohorts without fitting models."""
import argparse
from functools import partial
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import polars as pl

from scripts.audit_history_features import audit as audit_history
from scripts.audit_itemstats_features import audit as audit_itemstats
from scripts.audit_item2item_features import audit as audit_item2item
from scripts.audit_weighted_item2item_features import audit as audit_weighted
from scripts.run_joint_oof import load_part
from src.common.feature_cache import cache_lock
from src.common.history import build_features as build_history
from src.common.itemstats import build_features as build_itemstats
from src.common.item2item import build_features as build_item2item
from src.common.weighted_item2item import build_features as build_weighted
from src.common.preparation import fingerprint, write_json
from src.common.train_selection import select_parts
from src.gbdt.prepared import BASE


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_cache(prepared, root, entries, *, excluded=None, exact_train=True):
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if not manifest.get('complete'):
        raise ValueError(f'incomplete cache: {root}')
    if excluded is not None and set(manifest.get('excluded_train_sources', [])) != excluded:
        raise ValueError(f'donor exclusions differ from block: {root}')
    identity = manifest.get('identity', {})
    if excluded is not None and set(identity.get('excluded_train_sources', [])) != excluded:
        raise ValueError(f'donor identity differs from block: {root}')
    for relative, stamp in identity.get('input_fingerprints', {}).items():
        if fingerprint(prepared / relative) != stamp:
            raise ValueError(f'cache donor or query input changed: {relative}')
    for name, stamp in manifest.get('artifacts', {}).items():
        if fingerprint(root / name) != stamp:
            raise ValueError(f'cache aggregate changed: {root / name}')
    if 'statistics_fingerprint' in manifest:
        if fingerprint(root / '_statistics.duckdb') != manifest['statistics_fingerprint']:
            raise ValueError('itemstats database changed')
    for split, parts in entries.items():
        saved = manifest['splits'][split]['parts']
        required = {entry['source'] for entry in parts}
        if not required <= set(saved) or ((split != 'train' or exact_train) and required != set(saved)):
            raise ValueError(f'cache coverage differs: {root}, {split}')
        for entry in parts:
            part = saved[entry['source']]
            source_stamp = entry['source_fingerprint'] if 'catalogue_sha256' in manifest else fingerprint(prepared / entry['candidate_path'])
            if (part['rows'] != entry['candidates']
                    or part['source'] != source_stamp
                    or part['fingerprint'] != fingerprint(root / part['path'])):
                raise ValueError(f'cache part changed: {root}, {entry["source"]}')
            if 'feeds_source' in part and part['feeds_source'] != fingerprint(prepared / entry['feed_path']):
                raise ValueError('cache history input changed')
    return manifest


def verify_block(prepared, source, block, shared):
    if not block.get('complete'):
        raise ValueError('incomplete block')
    cohort = block['cohort']
    selected = select_parts(source, len(cohort['sources']), cohort['start'])
    if [entry['source'] for entry in selected] != cohort['sources']:
        raise ValueError('block train sources changed')
    entries = {'train': selected, 'val': source['splits']['val']['parts'], 'test': source['splits']['test']['parts']}
    for root_value in block['feature_roots']:
        root = Path(root_value)
        evidence = next((info for info in shared.values() if info['root'] == str(root)), None)
        if evidence is not None:
            if digest(root / 'manifest.json') != evidence['manifest_sha256']:
                raise ValueError('shared feature manifest changed')
            verify_cache(prepared, root, entries, exact_train=False)
            continue
        evidence = block['audits'][root.name]
        if (digest(evidence['path']) != evidence['sha256']
                or digest(root / 'manifest.json') != evidence['manifest_sha256']):
            raise ValueError('block feature audit changed')
        verify_cache(prepared, root, entries, excluded=None if root.name == 'history' else set(cohort['sources']))
    return selected


def prepare_blocks(prepared, raw, feature_dir, output, *, starts=(0, 32, 64), train_parts=32,
                   cosine=Path('data/features/cosine'), infeed=Path('data/features/infeed'),
                   oof_out=Path('runs/dima-blocks-oof'), weighted=False, memory_mb=1536,
                   threads=1, plan_only=False, resume=False):
    prepared, raw, feature_dir, output, cosine, infeed, oof_out = map(
        lambda path: Path(path).resolve(), (prepared, raw, feature_dir, output, cosine, infeed, oof_out))
    starts = list(starts)
    if not starts or len(set(starts)) != len(starts) or memory_mb < 128 or threads < 1:
        raise ValueError('invalid block or resource configuration')
    destinations = (feature_dir, output, oof_out)
    for destination in destinations:
        for source_root in (prepared, raw, cosine, infeed):
            if destination == source_root or destination in source_root.parents or source_root in destination.parents:
                raise ValueError('block outputs must be separate from input directories')
    if any(a == b or a in b.parents or b in a.parents for i, a in enumerate(destinations) for b in destinations[i + 1:]):
        raise ValueError('block feature, report and model directories must be separate')
    source_path = prepared / 'manifest.json'
    source_sha = digest(source_path)
    source = json.loads(source_path.read_text(encoding='utf-8'))
    if not source.get('complete'):
        raise ValueError('complete prepared data required')
    cohorts, seen_sources, seen_feeds = [], set(), set()
    for start in starts:
        selected = select_parts(source, train_parts, start)
        names = {entry['source'] for entry in selected}
        if names & seen_sources or len(names) == len(source['splits']['train']['parts']):
            raise ValueError('training blocks overlap or leave no donor shards')
        seen_sources.update(names)
        feeds = set()
        for entry in selected:
            ids = pl.read_parquet(prepared / entry['feed_path'], columns=['feed_id'])['feed_id']
            values = ids.to_list()
            if ids.null_count() or ids.is_duplicated().any() or feeds.intersection(values):
                raise ValueError('duplicate or null training feed identifiers')
            feeds.update(values)
        if feeds & seen_feeds:
            raise ValueError('feed identifiers overlap between train blocks')
        seen_feeds.update(feeds)
        cohorts.append({'start': start, 'stop_exclusive': start + train_parts,
                        'sources': [entry['source'] for entry in selected], 'feeds': len(feeds),
                        'candidate_rows': sum(entry['candidates'] for entry in selected)})
    del seen_feeds
    entries = {'train': [entry for entry in source['splits']['train']['parts'] if entry['source'] in seen_sources],
               'val': source['splits']['val']['parts'], 'test': source['splits']['test']['parts']}
    shared = {}
    for name, root in (('cosine', cosine), ('infeed', infeed)):
        verify_cache(prepared, root, entries, exact_train=False)
        shared[name] = {'root': str(root), 'manifest_sha256': digest(root / 'manifest.json')}
    code_paths = ['src/common/train_selection.py', 'src/common/feature_cache.py', 'src/common/history.py',
                  'src/common/itemstats.py', 'src/common/itemstats_lookup.py', 'src/common/item2item.py', 'src/common/weighted_item2item.py',
                  'src/common/similarity.py', 'src/common/infeed.py', 'scripts/feature_audit.py', 'scripts/prepare_train_blocks.py',
                  'scripts/audit_history_features.py', 'scripts/audit_itemstats_features.py',
                  'scripts/audit_item2item_features.py', 'scripts/audit_weighted_item2item_features.py']
    identity = {'prepared': str(prepared), 'prepared_manifest_sha256': source_sha, 'cohorts': cohorts, 'weighted': weighted,
                'feature_dir': str(feature_dir), 'raw': str(raw), 'oof_out': str(oof_out), 'shared': shared,
                'code_sha256': {name: digest(name) for name in code_paths}}
    with cache_lock(output):
        report_path = output / 'run.json'
        report = {'identity': identity, 'complete': False, 'blocks': {}, 'started_unix': time.time()}
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding='utf-8'))
            if not resume or report['identity'] != identity:
                raise ValueError('existing block preparation requires matching inputs and --resume')

        def checkpoint(stage):
            report.pop('error', None)
            report.update(stage=stage, complete=False, status='running',
                          elapsed_seconds=round(time.time() - report['started_unix'], 2))
            write_json(report_path, report)
            print(stage, flush=True)

        try:
            if report.get('complete'):
                for block in report['blocks'].values():
                    verify_block(prepared, source, block, shared)
                return report
            for cohort in cohorts:
                start = cohort['start']
                tag = f'block-{start:03d}-{cohort["stop_exclusive"]:03d}'
                block_root = feature_dir / tag
                roots = {name: block_root / name for name in ('history', 'itemstats', 'item2item')}
                if weighted:
                    roots['weighted-item2item'] = block_root / 'weighted-item2item'
                feature_roots = [cosine, infeed, *roots.values()]
                argv = ['uv', 'run', '--locked', 'python', '-m', 'scripts.run_joint_oof',
                        '--prepared', str(prepared), '--raw', str(raw), '--train-parts', str(train_parts),
                        '--train-start', str(start), '--folds', '4', '--iterations', '92', '--depth', '6',
                        '--learning-rate', '0.1', '--seed', '42', '--threads', '2', '--memory-mb', '512',
                        '--out', str(oof_out / tag), '--resume']
                for root in feature_roots:
                    argv.extend(('--feature-root', str(root)))
                block = report['blocks'].setdefault(tag, {'cohort': cohort, 'audits': {}})
                block.update(feature_roots=[str(root) for root in feature_roots], oof_argv=argv,
                             oof_powershell='& ' + ' '.join("'" + argument.replace("'", "''") + "'" for argument in argv),
                             donor_parts=len(source['splits']['train']['parts']) - train_parts)
                if plan_only:
                    continue
                selected = select_parts(source, train_parts, start)
                block_entries = {**entries, 'train': selected}
                options = {'train_parts': train_parts, 'train_start': start, 'resume': True}
                steps = [
                    ('history', partial(build_history, prepared, roots['history'], **options),
                     partial(audit_history, prepared, roots['history'])),
                    ('itemstats', partial(build_itemstats, prepared, roots['itemstats'], memory_mb=memory_mb,
                                          threads=threads, **options), partial(audit_itemstats, prepared, roots['itemstats'])),
                    ('item2item', partial(build_item2item, prepared, roots['item2item'], memory_mb=memory_mb, **options),
                     partial(audit_item2item, prepared, roots['item2item']))]
                if weighted:
                    steps.append(('weighted-item2item', partial(build_weighted, prepared, roots['item2item'],
                        roots['weighted-item2item'], **options), partial(audit_weighted, prepared,
                        roots['weighted-item2item'], roots['item2item'], output / tag / 'item2item-audit.json')))
                with cache_lock(block_root):
                    for name, build, audit in steps:
                        checkpoint(f'{tag}/{name}/build')
                        evidence = block['audits'].get(name)
                        if evidence is None:
                            build()
                        verify_cache(prepared, roots[name], block_entries,
                                     excluded=None if name == 'history' else set(cohort['sources']))
                        manifest_sha = digest(roots[name] / 'manifest.json')
                        audit_path = output / tag / f'{name}-audit.json'
                        if (evidence is None or evidence['manifest_sha256'] != manifest_sha
                                or not audit_path.exists() or digest(audit_path) != evidence['sha256']):
                            checkpoint(f'{tag}/{name}/audit')
                            audited = audit(output=audit_path)
                            if not audited.get('complete') or audited['manifest_sha256'] != manifest_sha:
                                raise ValueError('audit does not match completed block cache')
                            block['audits'][name] = {'path': str(audit_path), 'sha256': digest(audit_path),
                                                    'manifest_sha256': manifest_sha}
                        checkpoint(f'{tag}/{name}/complete')
                        gc.collect()
                checkpoint(f'{tag}/integration')
                fields = list(BASE)
                for root in feature_roots:
                    fields.extend(json.loads((root / 'manifest.json').read_text())['features'])
                if len(fields) != (121 if weighted else 109) or len(fields) != len(set(fields)):
                    raise ValueError('unexpected joint feature contract')
                loaded = {}
                for split, parts in block_entries.items():
                    frame = load_part(prepared, parts[0], split, feature_roots)
                    if frame.height != parts[0]['candidates'] or not np.isfinite(frame.select(fields).to_numpy()).all():
                        raise ValueError('block feature integration failed')
                    loaded[split] = frame.height
                    del frame
                block.update(complete=True, model_features=len(fields), integration_rows=loaded)
                checkpoint(f'{tag}/complete')
            if digest(source_path) != source_sha or any(digest(Path(info['root']) / 'manifest.json') != info['manifest_sha256']
                                                       for info in shared.values()):
                raise ValueError('shared inputs changed during block preparation')
            report.update(complete=not plan_only, status='planned' if plan_only else 'complete',
                          stage='planned' if plan_only else 'complete',
                          elapsed_seconds=round(time.time() - report['started_unix'], 2))
            write_json(report_path, report)
            return report
        except BaseException as error:
            report.update(status='failed', complete=False, error=repr(error))
            write_json(report_path, report)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, default=Path('data/prepared'))
    parser.add_argument('--raw', type=Path, default=Path('competition'))
    parser.add_argument('--feature-dir', type=Path, default=Path('data/features/train-blocks'))
    parser.add_argument('--out', dest='output', type=Path, required=True)
    parser.add_argument('--cosine', type=Path, default=Path('data/features/cosine'))
    parser.add_argument('--infeed', type=Path, default=Path('data/features/infeed'))
    parser.add_argument('--oof-out', type=Path, default=Path('runs/dima-blocks-oof'))
    parser.add_argument('--starts', type=int, nargs='+', default=[0, 32, 64])
    parser.add_argument('--train-parts', type=int, default=32)
    parser.add_argument('--weighted-item2item', dest='weighted', action='store_true')
    parser.add_argument('--memory-mb', type=int, default=1536)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    report = prepare_blocks(**vars(args))
    print(json.dumps({'status': report['status'], 'blocks': list(report['blocks'])}, indent=2))


if __name__ == '__main__':
    main()
