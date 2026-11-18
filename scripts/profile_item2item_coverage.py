"""Measure candidate index coverage and nonzero weighted co-history across splits."""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from scripts.run_joint_oof import digest
from src.common.item2item import ItemHistoryIndex
from src.common.preparation import fingerprint, write_json


CHANNELS = ['all', 'evt2', 'evt10']


def profile(cache, parent):
    manifest = json.loads((cache / 'manifest.json').read_bytes())
    donor = json.loads((parent / 'manifest.json').read_bytes())
    if (not manifest['complete'] or not donor['complete']
            or manifest['identity']['extra']['parent_identity'] != donor['identity']
            or manifest['identity']['extra']['parent_artifacts'] != donor['artifacts']):
        raise ValueError('weighted cache and donor index must have matching identities')
    splits = {}
    with ItemHistoryIndex(parent) as index:
        for split, record in manifest['splits'].items():
            frames = []
            for part in record['parts'].values():
                path = cache / part['path']
                if fingerprint(path) != part['fingerprint']:
                    raise ValueError(f'weighted feature part changed: {path}')
                names = [f'wi2i_{channel}_max_50' for channel in CHANNELS]
                frame = pl.read_parquet(path, columns=['feed_id', 'item_id', *names])
                ids = frame['item_id'].to_numpy()
                positions = np.searchsorted(index.items, ids)
                known = positions < len(index.items)
                known[known] = index.items[positions[known]] == ids[known]
                if ((frame[names[0]].to_numpy() > 0) & ~known).any():
                    raise ValueError('positive co-history for candidate outside donor index')
                frames.append(frame.with_columns(pl.Series('known', known)).group_by('feed_id').agg(
                    pl.len().alias('rows'), pl.col('known').sum(),
                    *[(pl.col(name) > 0).sum().alias(c) for c, name in zip(CHANNELS, names)]))
            feeds = pl.concat(frames)
            if feeds['feed_id'].is_duplicated().any():
                raise ValueError('feed appears in multiple cache parts')
            rows = int(feeds['rows'].sum())
            known = int(feeds['known'].sum())
            supported = int(feeds['all'].sum())
            splits[split] = dict(feeds=feeds.height, rows=rows, absent_from_index=rows-known,
                                 known_no_match=known-supported, known_match=supported,
                                 match_given_known=supported/known if known else None,
                                 channels={c: dict(candidates=int(feeds[c].sum()),
                                                   feeds=int((feeds[c] > 0).sum())) for c in CHANNELS})
    return dict(cache_sha256=digest(cache / 'manifest.json'), donor_sha256=digest(parent / 'manifest.json'),
                support='positive weighted maximum over the first 50 history events', splits=splits)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    write_json(args.out, profile(args.cache, args.parent))
