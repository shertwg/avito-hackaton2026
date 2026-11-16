"""Within-feed ranks of cosine and donor item CTR features."""
import hashlib
import inspect
import json
from pathlib import Path

import polars as pl

from src.common.feature_cache import KEY, build_cache, selected_parts
from src.common.infeed import add_infeed_stats
from src.common.preparation import fingerprint

INPUTS = ['cos_user', 'cos_hist_max_10', 'cos_hist_max_50', 'is_item_ctr']


def feature_names():
    return [f'fr_{name}' for name in INPUTS]


def domains():
    return {name: (0, 1) for name in feature_names()}


def dependency_identity(prepared, cosine, itemstats, train_parts):
    source = json.loads((Path(prepared) / 'manifest.json').read_text())
    requested = {'train': selected_parts(source, train_parts),
                 **{split: source['splits'][split]['parts'] for split in ('val', 'test')}}
    result = {}
    for name, root in (('cosine', cosine), ('itemstats', itemstats)):
        manifest = json.loads((Path(root) / 'manifest.json').read_text())
        if not manifest.get('complete'):
            raise ValueError('rank dependency cache incomplete')
        parts = {}
        for split, entries in requested.items():
            parts[split] = {}
            for entry in entries:
                part = manifest['splits'][split]['parts'].get(entry['source'])
                if part is None:
                    raise ValueError('rank dependency does not cover requested cohort')
                parts[split][entry['source']] = {
                    key: part[key] for key in ('path', 'rows', 'source', 'fingerprint', 'feeds_source') if key in part}
        result[name] = {'parts': parts, 'value_signature': manifest.get('value_signature'),
                        'identity': manifest.get('identity'), 'catalogue_sha256': manifest.get('catalogue_sha256')}
    result['rank_math_sha256'] = hashlib.sha256(inspect.getsource(add_infeed_stats).encode()).hexdigest()
    return result


def source_frame(prepared, root, split, entry, columns):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    part = manifest['splits'][split]['parts'].get(entry['source'])
    if part is None or fingerprint(root / part['path']) != part['fingerprint']:
        raise ValueError('rank source part missing or changed')
    expected = (entry['source_fingerprint'] if manifest.get('feature_group', 'cosine') == 'cosine'
                else fingerprint(Path(prepared) / entry['candidate_path']))
    if part['source'] != expected:
        raise ValueError('rank source refers to different candidates')
    return pl.read_parquet(root / part['path'], columns=[*KEY, 'item_id', *columns])


def build_features(prepared, cosine, itemstats, output, *, train_parts=8, resume=False):
    prepared, cosine, itemstats = map(Path, (prepared, cosine, itemstats))
    source = json.loads((prepared / 'manifest.json').read_text())
    selected = {entry['source'] for entry in selected_parts(source, train_parts)}
    statistics = json.loads((itemstats / 'manifest.json').read_text())
    if set(statistics['excluded_train_sources']) != selected:
        raise ValueError('itemstats exclusion cohort differs from requested rank cohort')

    def build_part(split, entry, state):
        frame = pl.read_parquet(prepared / entry['candidate_path'], columns=[*KEY, 'item_id', 'category_id'])
        for root, columns in ((cosine, INPUTS[:3]), (itemstats, INPUTS[3:])):
            values = source_frame(prepared, root, split, entry, columns)
            if values.height != frame.height:
                raise ValueError('rank source row count differs')
            frame = frame.join(values, on=[*KEY, 'item_id'], how='left', validate='1:1')
        if frame.select(INPUTS).null_count().to_numpy().any() or not all(frame[name].is_finite().all() for name in INPUTS):
            raise ValueError('invalid source values for ranks')
        ranked = add_infeed_stats(frame.lazy(), columns=INPUTS)
        return ranked.select(*KEY, 'item_id', *[
            pl.col(f'{name}_rank').alias(f'fr_{name}') for name in INPUTS]).sort('feed_id', 'candidate_index').collect()

    return build_cache(prepared, output, train_parts=train_parts, group='feature_ranks', names=feature_names(),
                       domains=domains(), producer=__file__, extra=dependency_identity(prepared, cosine, itemstats, train_parts),
                       build_part=build_part, resume=resume, excluded=selected,
                       verify_extra=lambda: dependency_identity(prepared, cosine, itemstats, train_parts))
