"""Within-feed deviations, standardized signals and cosine/CTR interactions."""
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

from src.common.feature_cache import KEY, build_cache, selected_parts
from src.common.feature_ranks import dependency_identity as rank_dependencies, source_frame

BASE_FIELDS = [f'f{i}' for i in range(1, 6)]
SIGNALS = ['cos_user', 'is_item_ctr', 'i2i_sum_10', 'i2i_max_10', 'i2i_sum_50', 'i2i_max_50']
INPUTS = [*BASE_FIELDS, *SIGNALS, 'cos_hist_max_50']


def feature_names():
    result = [f'fc_{name}_{operation}' for name in BASE_FIELDS for operation in ('gap_max', 'gap_mean')]
    for name in SIGNALS:
        operations = ('z', 'rank', 'gap_max', 'gap_mean') if name.startswith('i2i_') else ('z', 'gap_max', 'gap_mean')
        result.extend(f'fc_{name}_{operation}' for operation in operations)
    return result + ['fc_cos_user_x_item_ctr', 'fc_cos_hist_max_50_x_item_ctr']


def domains():
    bound = float(np.finfo(np.float32).max)
    return {name: (0, 1) if name.endswith('_rank') else (-1, 1) if '_x_' in name else (-bound, bound)
            for name in feature_names()}


def dependency_identity(prepared, cosine, itemstats, item2item, train_parts):
    result = rank_dependencies(prepared, cosine, itemstats, train_parts)
    source = json.loads((Path(prepared) / 'manifest.json').read_text())
    parent = json.loads((Path(item2item) / 'manifest.json').read_text())
    selected = {p['source'] for p in selected_parts(source, train_parts)}
    for root in (itemstats, item2item):
        manifest = json.loads((Path(root) / 'manifest.json').read_text())
        if not manifest.get('complete') or set(manifest['excluded_train_sources']) != selected:
            raise ValueError('feed context donor cohort differs')
    parts = {}
    for split, info in source['splits'].items():
        requested = selected if split == 'train' else {p['source'] for p in info['parts']}
        parts[split] = {name: {k: value for k, value in parent['splits'][split]['parts'][name].items()
            if k in ('path', 'rows', 'source', 'fingerprint', 'feeds_source')} for name in sorted(requested)}
    result['item2item'] = {'identity': parent['identity'], 'parts': parts}
    result['reader_sha256'] = hashlib.sha256(Path(__file__).with_name('feature_ranks.py').read_bytes()).hexdigest()
    result['formula'] = 'sample std; average rank/n; value-max and value-mean; float64 intermediates'
    return result


def add_fields(frame):
    if frame.select(INPUTS).null_count().to_numpy().any() or not all(frame[name].is_finite().all() for name in INPUTS):
        raise ValueError('feed context inputs must be finite and non-null')
    if not frame['is_item_ctr'].is_between(0, 1).all():
        raise ValueError('item CTR outside [0,1]')
    expressions = []
    for name in [*BASE_FIELDS, *SIGNALS]:
        value = pl.col(name).cast(pl.Float64)
        mean, maximum = value.mean().over('feed_id'), value.max().over('feed_id')
        expressions.extend(((value - maximum).alias(f'fc_{name}_gap_max'), (value - mean).alias(f'fc_{name}_gap_mean')))
        if name in SIGNALS:
            deviation = value.std(ddof=1).over('feed_id')
            expressions.append(pl.when(deviation > 0).then((value - mean) / deviation).otherwise(0.).alias(f'fc_{name}_z'))
            if name.startswith('i2i_'):
                expressions.append((value.rank('average').over('feed_id') / pl.len().over('feed_id')).alias(f'fc_{name}_rank'))
    expressions.extend((
        (pl.col('cos_user').cast(pl.Float64) * pl.col('is_item_ctr')).alias('fc_cos_user_x_item_ctr'),
        (pl.col('cos_hist_max_50').cast(pl.Float64) * pl.col('is_item_ctr')).alias('fc_cos_hist_max_50_x_item_ctr')))
    bound = float(np.finfo(np.float32).max)
    return frame.with_columns(expressions).select(*KEY, 'item_id', *[
        pl.col(name).clip(-bound, bound).cast(pl.Float32) for name in feature_names()]).sort('feed_id', 'candidate_index')


def input_frame(prepared, cosine, itemstats, item2item, split, entry):
    frame = pl.read_parquet(Path(prepared) / entry['candidate_path'], columns=[*KEY, 'item_id', *BASE_FIELDS])
    for root, fields in ((cosine, ['cos_user', 'cos_hist_max_50']), (itemstats, ['is_item_ctr']), (item2item, SIGNALS[2:])):
        values = source_frame(prepared, root, split, entry, fields)
        if values.height != frame.height:
            raise ValueError('feed context source row count differs')
        frame = frame.join(values, on=[*KEY, 'item_id'], how='left', validate='1:1')
    return frame.sort('feed_id', 'candidate_index')


def build_features(prepared, cosine, itemstats, item2item, output, *, train_parts=8, resume=False):
    prepared, cosine, itemstats, item2item = map(Path, (prepared, cosine, itemstats, item2item))
    extra = dependency_identity(prepared, cosine, itemstats, item2item, train_parts)
    selected = set(json.loads((itemstats / 'manifest.json').read_text())['excluded_train_sources'])

    def build_part(split, entry, state):
        return add_fields(input_frame(prepared, cosine, itemstats, item2item, split, entry))

    return build_cache(prepared, output, train_parts=train_parts, group='feed_context', names=feature_names(),
        domains=domains(), producer=__file__, extra=extra, build_part=build_part, excluded=selected, resume=resume,
        verify_extra=lambda: dependency_identity(prepared, cosine, itemstats, item2item, train_parts))
