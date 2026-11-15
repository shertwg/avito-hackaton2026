"""Small competition-format input shared by additional-feature regression tests."""
import json
from pathlib import Path

import numpy as np
import polars as pl

from eval.make_synthetic import make_split
from src.common.feature_cache import KEY, selected_parts
from src.common.preparation import fingerprint, prepare_dataset, write_json


def make_prepared(root):
    root = Path(root)
    raw, prepared = root / 'raw', root / 'prepared'
    rng = np.random.default_rng(57)
    for index, split in enumerate(('train', 'val', 'test')):
        make_split(raw, split, 8, rng, index * 100, split == 'train', split != 'test')
    path = raw / 'train/impressions/part-00000.parquet'
    frame = pl.read_parquet(path)
    for index in range(4):
        frame.slice(index * 2, 2).write_parquet(path.with_name(f'part-{index:05d}.parquet'))
    test = pl.read_parquet(raw / 'test/impressions/part-00000.parquet')
    test.select('feed_id', 'item_id').explode('item_id').cast(
        {'feed_id': pl.Int32, 'item_id': pl.Int32}).with_columns(
            pl.lit(0., dtype=pl.Float32).alias('score')).write_parquet(raw / 'sample_submission.parquet')
    prepare_dataset(raw, prepared, root / 'eda')
    source = json.loads((prepared / 'manifest.json').read_text())
    return raw, prepared, source


def rank_sources(prepared, source, root):
    roots = [Path(root) / 'cosine', Path(root) / 'itemstats']
    columns = [('cos_user', 'cos_hist_max_10', 'cos_hist_max_50'), ('is_item_ctr',)]
    selected = {entry['source'] for entry in selected_parts(source, 2)}
    for kind, output, names in zip(('cosine', 'itemstats'), roots, columns):
        manifest = {'complete': True, 'feature_group': kind, 'features': list(names), 'splits': {}}
        if kind == 'itemstats':
            manifest['excluded_train_sources'] = sorted(selected)
        for split, info in source['splits'].items():
            saved = {}
            for entry in info['parts']:
                if kind == 'itemstats' and split == 'train' and entry['source'] not in selected:
                    continue
                frame = pl.read_parquet(Path(prepared) / entry['candidate_path'], columns=[*KEY, 'item_id'])
                index = np.arange(frame.height)
                values = {'cos_user': index % 3 - 1., 'cos_hist_max_10': (index % 3 - 1.) / 4.,
                          'cos_hist_max_50': np.full(frame.height, 0.5), 'is_item_ctr': (index % 2) / 10.}
                frame = frame.with_columns([pl.Series(name, values[name], dtype=pl.Float32) for name in names])
                path = output / split / entry['source']
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.write_parquet(path)
                saved[entry['source']] = {'path': path.relative_to(output).as_posix(),
                                          'rows': frame.height, 'fingerprint': fingerprint(path),
                                          'source': entry['source_fingerprint'] if kind == 'cosine'
                                          else fingerprint(Path(prepared) / entry['candidate_path'])}
            manifest['splits'][split] = {'parts': saved}
        write_json(output / 'manifest.json', manifest)
    return roots
