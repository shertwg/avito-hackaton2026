"""Independently audit category co-occurrence using bitsets of donor feeds."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import polars as pl

from src.common.category_cooc import KEY, RECENT, feature_names, validate_features
from src.common.preparation import fingerprint, write_json
from src.common.resources import memory_snapshot


def audit(prepared, cache, output):
    prepared, cache, output = map(Path, (prepared, cache, output))
    started = time.monotonic()
    source_bytes = (prepared / 'manifest.json').read_bytes()
    source = json.loads(source_bytes)
    manifest_bytes = (cache / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    identity = manifest['identity']
    if not source.get('complete') or not manifest.get('complete'):
        raise ValueError('complete source and cache required')
    if manifest.get('feature_group') != 'category_cooc' or manifest['features'] != feature_names():
        raise ValueError('unexpected feature group')
    if set(manifest['splits']) != {'train', 'val', 'test'}:
        raise ValueError('incomplete split coverage')
    if identity['prepared_manifest_sha256'] != hashlib.sha256(source_bytes).hexdigest():
        raise ValueError('prepared manifest changed')
    donors = set(identity['donor_sources'])
    excluded = set(manifest['excluded_train_sources'])
    all_train = {entry['source'] for entry in source['splits']['train']['parts']}
    if (donors & excluded or donors | excluded != all_train
            or excluded != set(identity['excluded_train_sources'])
            or set(manifest['splits']['train']['parts']) != excluded):
        raise ValueError('donor exclusion differs')
    for relative, stamp in identity['input_fingerprints'].items():
        if fingerprint(prepared / relative) != stamp:
            raise ValueError('input changed')
    signatures = [bytearray() for _ in range(256)]
    donor_feeds = 0
    for number, entry in enumerate(source['splits']['train']['parts'], 1):
        if entry['source'] not in donors:
            continue
        frame = pl.read_parquet(prepared / entry['feed_path'], columns=['history__category_id'])
        packed = np.zeros((256, (frame.height + 7) // 8), dtype=np.uint8)
        for index, (history,) in enumerate(frame.iter_rows()):
            if history is None or None in history:
                raise ValueError('null donor history')
            for category in set(history):
                packed[category + 128, index // 8] |= 1 << (index % 8)
        for category in range(256):
            signatures[category].extend(packed[category].tobytes())
        donor_feeds += frame.height
        if number % 128 == 0:
            print(f'category cooc audit donor source: {number}', flush=True)
    bits = [int.from_bytes(values, 'little') for values in signatures]
    support = [value.bit_count() for value in bits]
    active = [index for index, count in enumerate(support) if count]
    reference = np.zeros((256, 256), dtype=np.int64)
    for left in active:
        for right in active:
            reference[left, right] = (bits[left] & bits[right]).bit_count()
    state_path = cache / '_counts.npz'
    if fingerprint(state_path) != manifest['statistics_fingerprint']:
        raise ValueError('count checkpoint changed')
    with np.load(state_path, allow_pickle=False) as state:
        if int(state['processed']) != len(donors) or not np.array_equal(state['counts'], reference):
            raise ValueError('independent donor count mismatch')
        expected_identity = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        if str(state['identity_sha256']) != expected_identity:
            raise ValueError('count checkpoint identity differs')
    report = {'complete': False, 'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
              'donor_parts': len(donors), 'donor_feeds': donor_feeds,
              'excluded_train_parts': len(excluded), 'active_categories': len(active),
              'independent_counts': 'all category pairs via intersections of donor-feed bitsets',
              'sampled_feeds': 0, 'sampled_candidate_rows': 0, 'bytes': 0, 'splits': {}}
    for split, group in manifest['splits'].items():
        entries = {entry['source']: entry for entry in source['splits'][split]['parts']}
        if split != 'train' and set(entries) != set(group['parts']):
            raise ValueError('incomplete validation or test coverage')
        rows = 0
        for name, part in group['parts'].items():
            entry = entries[name]
            if part['rows'] != entry['candidates']:
                raise ValueError('manifest row count differs')
            path = cache / part['path']
            if (fingerprint(path) != part['fingerprint']
                    or fingerprint(prepared / entry['candidate_path']) != part['source']
                    or fingerprint(prepared / entry['feed_path']) != part['feeds_source']):
                raise ValueError('feature or source fingerprint differs')
            candidates = pl.read_parquet(prepared / entry['candidate_path'],
                                         columns=[*KEY, 'item_id', 'category_id']).sort('feed_id', 'candidate_index')
            features = pl.read_parquet(path)
            validate_features(features, candidates.height)
            if not features.select(*KEY, 'item_id').equals(candidates.select(*KEY, 'item_id')):
                raise ValueError('candidate keys differ')
            feeds = pl.read_parquet(prepared / entry['feed_path'], columns=['feed_id', 'history__category_id'])
            positions = sorted({0, feeds.height // 2, feeds.height - 1}) if feeds.height else []
            histories = {feeds['feed_id'][index]: feeds['history__category_id'][index].to_list() for index in positions}
            sample_candidates = candidates.filter(pl.col('feed_id').is_in(list(histories)))
            sample_features = features.filter(pl.col('feed_id').is_in(list(histories)))
            expected = []
            for row in sample_candidates.iter_rows(named=True):
                left = row['category_id'] + 128
                values = []
                for recent in RECENT:
                    similarities = []
                    for category in histories[row['feed_id']][:recent]:
                        right = category + 128
                        denominator = math.sqrt(support[left] * support[right])
                        similarities.append(reference[left, right] / denominator if denominator else 0.)
                    values.extend((sum(similarities), max(similarities, default=0.)))
                expected.append(values)
            expected = np.asarray(expected, dtype=np.float32).reshape(-1, len(feature_names()))
            if not np.allclose(sample_features.select(feature_names()).to_numpy(), expected, atol=2e-6, rtol=2e-6):
                raise ValueError('independent category co-occurrence formula mismatch')
            report['sampled_feeds'] += len(histories)
            report['sampled_candidate_rows'] += sample_candidates.height
            report['bytes'] += path.stat().st_size
            rows += features.height
        report['splits'][split] = {'parts': len(group['parts']), 'rows': rows}
    for relative, stamp in identity['input_fingerprints'].items():
        if fingerprint(prepared / relative) != stamp:
            raise ValueError('input changed during audit')
    if ((cache / 'manifest.json').read_bytes() != manifest_bytes
            or (prepared / 'manifest.json').read_bytes() != source_bytes):
        raise ValueError('manifest changed during audit')
    report.update(complete=True, duration_seconds=round(time.monotonic() - started, 2),
                  memory=memory_snapshot(), audit_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(output, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, default=Path('data/prepared'))
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.prepared, args.cache, args.out), indent=2))
