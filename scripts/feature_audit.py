"""Structural checks and sampled independent formulas for new feature groups."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import polars as pl

from src.common.feature_cache import KEY, selected_parts, validate_frame
from src.common.preparation import fingerprint, write_json
from src.common.resources import memory_snapshot


def sample_candidates(frame):
    ids = frame['feed_id'].unique(maintain_order=True)
    positions = sorted({0, len(ids) // 2, len(ids) - 1}) if len(ids) else []
    return frame.filter(pl.col('feed_id').is_in([ids[index] for index in positions]))


def audit_cache(prepared, cache, output, *, group, names, domains, reference,
                producer, candidate_columns=(), feed_columns=(), verify_extra=None, train_parts=None):
    prepared, cache, output = map(Path, (prepared, cache, output))
    started = time.monotonic()
    source_bytes = (prepared / 'manifest.json').read_bytes()
    manifest_bytes = (cache / 'manifest.json').read_bytes()
    source, manifest = json.loads(source_bytes), json.loads(manifest_bytes)
    if not source.get('complete') or not manifest.get('complete'):
        raise ValueError('complete source and feature cache required')
    if (manifest.get('feature_group') != group or manifest['features'] != names
            or manifest.get('key') != KEY
            or manifest['domains'] != {name: list(limits) for name, limits in domains.items()}):
        raise ValueError('feature contract differs')
    identity = manifest['identity']
    if identity['prepared_manifest_sha256'] != hashlib.sha256(source_bytes).hexdigest():
        raise ValueError('prepared manifest differs')
    if verify_extra and verify_extra() != identity['extra']:
        raise ValueError('dependency identity differs')
    if set(manifest['splits']) != {'train', 'val', 'test'}:
        raise ValueError('incomplete split coverage')
    if 'excluded_train_sources' in manifest:
        if (set(manifest['excluded_train_sources']) != set(identity['excluded_train_sources'])
                or set(manifest['splits']['train']['parts']) != set(manifest['excluded_train_sources'])):
            raise ValueError('donor exclusion differs')
    for relative, stamp in identity['input_fingerprints'].items():
        if fingerprint(prepared / relative) != stamp:
            raise ValueError('prepared input changed')
    for name, stamp in manifest.get('artifacts', {}).items():
        if fingerprint(cache / name) != stamp:
            raise ValueError('aggregate artifact changed')
    report = {'complete': False, 'group': group, 'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
              'sampled_feeds': 0, 'sampled_candidate_rows': 0, 'bytes': 0, 'splits': {},
              'scope': 'all audited rows: keys, schema, domains and fingerprints; independent formulas on three feeds per part',
              'formula_atol': 2e-6, 'formula_rtol': 2e-6}
    for split, info in manifest['splits'].items():
        sources = {entry['source']: entry for entry in source['splits'][split]['parts']}
        parts = info['parts']
        if split == 'train' and train_parts is not None:
            expected = {entry['source'] for entry in selected_parts(source, train_parts)}
            if not expected <= set(parts):
                raise ValueError('requested training cohort not covered')
            parts = {name: parts[name] for name in sorted(expected)}
        if split != 'train' and set(parts) != set(sources):
            raise ValueError('validation or test coverage differs')
        rows = 0
        for number, (name, part) in enumerate(parts.items(), 1):
            entry = sources[name]
            path = cache / part['path']
            if (part['rows'] != entry['candidates'] or fingerprint(path) != part['fingerprint']
                    or fingerprint(prepared / entry['candidate_path']) != part['source']
                    or fingerprint(prepared / entry['feed_path']) != part['feeds_source']):
                raise ValueError('part source or feature fingerprint differs')
            columns = list(dict.fromkeys([*KEY, 'item_id', *candidate_columns]))
            candidates = pl.read_parquet(prepared / entry['candidate_path'], columns=columns).sort('feed_id', 'candidate_index')
            features = pl.read_parquet(path)
            validate_frame(features, names, candidates.height, domains)
            if not features.select(*KEY, 'item_id').equals(candidates.select(*KEY, 'item_id')):
                raise ValueError('occurrence keys differ')
            sample = sample_candidates(candidates)
            feeds = None
            if feed_columns:
                feeds = pl.read_parquet(prepared / entry['feed_path'], columns=['feed_id', *feed_columns])
                feeds = feeds.filter(pl.col('feed_id').is_in(sample['feed_id'].unique()))
            expected_values = reference(split, sample, feeds, entry)
            actual = features.filter(pl.col('feed_id').is_in(sample['feed_id'].unique())).select(names).to_numpy()
            if not np.allclose(actual, expected_values, atol=2e-6, rtol=2e-6):
                raise ValueError(f'independent {group} formula mismatch')
            rows += features.height
            report['sampled_feeds'] += sample['feed_id'].n_unique()
            report['sampled_candidate_rows'] += sample.height
            report['bytes'] += path.stat().st_size
            if number == len(parts):
                print(f'audit {group} {split}: {number} parts complete', flush=True)
        report['splits'][split] = {'parts': len(parts), 'rows': rows}
    if ((prepared / 'manifest.json').read_bytes() != source_bytes
            or (cache / 'manifest.json').read_bytes() != manifest_bytes
            or (verify_extra and verify_extra() != identity['extra'])):
        raise ValueError('source identity changed during audit')
    for relative, stamp in identity['input_fingerprints'].items():
        if fingerprint(prepared / relative) != stamp:
            raise ValueError('input changed during audit')
    report.update(complete=True, duration_seconds=round(time.monotonic() - started, 2), memory=memory_snapshot(),
                  audit_sha256=hashlib.sha256(Path(producer).read_bytes()).hexdigest(),
                  shared_audit_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(output, report)
    return report
