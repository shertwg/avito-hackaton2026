import json
import unittest

import numpy as np
import polars as pl

from scripts.audit_item2item_features import audit as audit_parent
from scripts.audit_weighted_item2item_features import audit
from src.common.item2item import ItemHistoryIndex
from src.common.weighted_item2item import build_feature_part, build_features, feature_names
from tests import test_item2item as fixtures


class WeightedItem2ItemTests(unittest.TestCase):
    def test_event_decay_normalization_and_audit(self):
        fixture = fixtures.Item2ItemTests('test_empty_donor_index')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        for path in fixture.prepared.glob('*/feeds/*.parquet'):
            feeds = pl.read_parquet(path)
            events = [[2 if i % 2 == 0 else 10 for i in range(len(history))]
                      for history in feeds['history__item_id'].to_list()]
            feeds.with_columns(pl.Series('history__event_type', events, dtype=pl.List(pl.Int8))).write_parquet(path)
        fixture.build('parent')
        root = fixture.root
        audit_parent(fixture.prepared, root / 'parent', root / 'parent-audit.json')
        build_features(fixture.prepared, root / 'parent', root / 'weighted', train_parts=2)
        frame = pl.read_parquet(root / 'weighted/train/part-00000.parquet')
        decay = 2. ** (-.1)
        self.assertAlmostEqual(frame['wi2i_all_mean_10'][0], (1 + .5 * decay) / (1 + decay), places=6)
        self.assertEqual(frame['wi2i_evt2_mean_10'][0], 1.)
        self.assertAlmostEqual(frame['wi2i_evt10_max_10'][0], .5 * decay, places=6)
        self.assertTrue((frame.tail(3).select(feature_names()).to_numpy() == 0).all())
        empty = pl.read_parquet(root / 'weighted/test/part-00000.parquet')
        self.assertTrue((empty.select(feature_names()).to_numpy() == 0).all())
        with ItemHistoryIndex(root / 'parent') as index:
            candidates = pl.read_parquet(fixture.prepared / 'train/candidates/part-00000.parquet')
            feeds = pl.read_parquet(fixture.prepared / 'train/feeds/part-00000.parquet')
            self.assertTrue(build_feature_part(candidates, feeds, index, candidate_batch=1).equals(frame))
            with self.assertRaisesRegex(ValueError, 'unaligned'):
                build_feature_part(candidates, feeds.with_columns(pl.lit([]).cast(pl.List(pl.Int8)).alias('history__event_type')), index)
        result = audit(fixture.prepared, root / 'weighted', root / 'parent', root / 'parent-audit.json', root / 'audit.json')
        self.assertTrue(result['complete'])
        self.assertEqual(len(feature_names()), 12)
        stamp = (root / 'weighted/train/part-00000.parquet').stat().st_mtime_ns
        build_features(fixture.prepared, root / 'parent', root / 'weighted', train_parts=2, resume=True)
        self.assertEqual((root / 'weighted/train/part-00000.parquet').stat().st_mtime_ns, stamp)
        with self.assertRaisesRegex(ValueError, 'cohort differs'):
            build_features(fixture.prepared, root / 'parent', root / 'wrong', train_parts=1)
        evidence = json.loads((root / 'parent-audit.json').read_text())
        evidence['manifest_sha256'] = 'invalid'
        (root / 'invalid.json').write_text(json.dumps(evidence))
        with self.assertRaisesRegex(ValueError, 'parent item2item audit'):
            audit(fixture.prepared, root / 'weighted', root / 'parent', root / 'invalid.json', root / 'no-audit.json')


if __name__ == '__main__':
    unittest.main()
