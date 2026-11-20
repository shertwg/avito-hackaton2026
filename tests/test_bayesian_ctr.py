import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import polars as pl

from scripts.audit_bayesian_ctr_features import audit
from src.common.bayesian_ctr import build_features, feature_names
from src.common.preparation import write_json


class BayesianCtrTests(unittest.TestCase):
    def test_hierarchical_prior_past_cutoffs_fresh_windows_and_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = root / 'prepared'
            source = {'complete': True, 'splits': {}}
            queries = [(11, 1, 1), (11, 1, 0), (99, 1, 0), (21, 2, 1)]
            parts = {'train': [(2, queries), (1, [(11, 1, 1), (11, 1, 0), (12, 1, 1), (21, 2, 0)]),
                               (10, [(11, 1, 0), (12, 1, 1), (21, 2, 1)]), (15, queries)],
                     'val': [(14, queries)], 'test': [(20, queries)]}
            for split, entries in parts.items():
                saved = []
                for number, (day, observations) in enumerate(entries):
                    name = f'part-{number:05d}.parquet'
                    feed_id = number + 100 * list(parts).index(split)
                    frame = pl.DataFrame([{'split': split, 'feed_id': feed_id, 'candidate_index': i,
                        'item_id': item, 'category_id': category, 'timestamp': day * 86400 + 3600, 'target': target}
                        for i, (item, category, target) in enumerate(observations)], schema_overrides={
                        'feed_id': pl.Int32, 'candidate_index': pl.UInt32, 'item_id': pl.Int32,
                        'category_id': pl.Int8, 'timestamp': pl.Int32, 'target': pl.Int8})
                    if split == 'test':
                        frame = frame.drop('target')
                    feeds = frame.select('feed_id', 'timestamp').unique()
                    candidate_path, feed_path = f'{split}/candidates/{name}', f'{split}/feeds/{name}'
                    for path, value in ((candidate_path, frame), (feed_path, feeds)):
                        target_path = prepared / path
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        value.write_parquet(target_path)
                    saved.append({'source': name, 'candidate_path': candidate_path, 'feed_path': feed_path, 'candidates': frame.height})
                source['splits'][split] = {'parts': saved}
            write_json(prepared / 'manifest.json', source)
            output = root / 'features'
            build_features(prepared, output, train_parts=2, smoothing=2., memory_mb=256)
            self.assertIn('donor train target counts', json.loads((output / 'manifest.json').read_text())['target_policy'])
            early = pl.read_parquet(output / 'train/part-00000.parquet')
            self.assertAlmostEqual(early['bc_category_ctr_all'][0], .6, places=6)
            self.assertAlmostEqual(early['bc_item_ctr_all'][0], .55, places=6)
            self.assertAlmostEqual(early['bc_item_category_delta_all'][0], -.05, places=6)
            self.assertEqual(early['bc_item_ctr_all'][0], early['bc_item_ctr_all'][1])
            self.assertAlmostEqual(early['bc_item_ctr_all'][2], .6, places=6)
            later = pl.read_parquet(output / 'val/part-00000.parquet')
            self.assertAlmostEqual(later['bc_item_ctr_3d'][0], 7 / 18, places=6)
            test = pl.read_parquet(output / 'test/part-00000.parquet')
            self.assertAlmostEqual(test['bc_item_ctr_3d'][0], 7 / 18, places=6)
            self.assertEqual(test['bc_donor_age_days'][0], 10.)
            self.assertTrue(audit(prepared, output, root / 'audit.json')['complete'])
            self.assertEqual(len(feature_names()), 14)
            stamp = (output / 'train/part-00000.parquet').stat().st_mtime_ns
            build_features(prepared, output, train_parts=2, smoothing=2., memory_mb=256, resume=True)
            self.assertIn('donor train target counts', json.loads((output / 'manifest.json').read_text())['target_policy'])
            self.assertEqual((output / 'train/part-00000.parquet').stat().st_mtime_ns, stamp)
            future = prepared / 'train/candidates/part-00002.parquet'
            pl.read_parquet(future).with_columns(pl.lit(1, dtype=pl.Int8).alias('target')).write_parquet(future)
            build_features(prepared, root / 'future-changed', train_parts=2, smoothing=2., memory_mb=256)
            self.assertTrue(early.equals(pl.read_parquet(root / 'future-changed/train/part-00000.parquet')))
            for entry in [source['splits']['train']['parts'][0], source['splits']['train']['parts'][3], source['splits']['val']['parts'][0]]:
                path = prepared / entry['candidate_path']
                pl.read_parquet(path).with_columns((1 - pl.col('target')).alias('target')).write_parquet(path)
            build_features(prepared, root / 'query-labels-changed', train_parts=2, smoothing=2., memory_mb=256)
            for path in (root / 'future-changed').glob('*/*.parquet'):
                other = root / 'query-labels-changed' / path.relative_to(root / 'future-changed')
                self.assertTrue(pl.read_parquet(path).equals(pl.read_parquet(other)))
            self.assertTrue(np.isfinite(test.select(feature_names()).to_numpy()).all())
            first_query = prepared / 'train/candidates/part-00000.parquet'
            pl.read_parquet(first_query).with_columns(pl.lit(3600, dtype=pl.Int32).alias('timestamp')).write_parquet(first_query)
            build_features(prepared, root / 'cold', train_parts=2, smoothing=2., memory_mb=256)
            cold = pl.read_parquet(root / 'cold/train/part-00000.parquet')
            self.assertTrue((cold.select(feature_names()).to_numpy() == 0).all())

            def interrupt(path, value):
                write_json(path, value)
                if path.name == '_stats.json' and len(value['files']) == 1:
                    raise RuntimeError('lookup publication interruption')

            interrupted = root / 'interrupted'
            with patch('src.common.bayesian_ctr.write_json', side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, 'lookup publication interruption'):
                    build_features(prepared, interrupted, train_parts=2, smoothing=2., memory_mb=256)
            first_lookup = interrupted / '_lookup_0.parquet'
            before = first_lookup.stat().st_mtime_ns
            build_features(prepared, interrupted, train_parts=2, smoothing=2., memory_mb=256, resume=True)
            self.assertEqual(first_lookup.stat().st_mtime_ns, before)
            self.assertTrue(audit(prepared, interrupted, root / 'resumed-audit.json')['complete'])
            with self.assertRaisesRegex(ValueError, 'separate donor'):
                build_features(prepared, root / 'no-donors', train_parts=4, smoothing=2., memory_mb=256)
            val_entry = source['splits']['val']['parts'][0]
            shuffled = pl.DataFrame({'split': ['val'] * 5, 'feed_id': [500, 100, 400, 200, 300],
                'candidate_index': [0] * 5, 'item_id': [99, 12, 98, 96, 97], 'category_id': [1] * 5,
                'timestamp': [14 * 86400 + 3600] * 5, 'target': [0] * 5}, schema_overrides={
                'feed_id': pl.Int32, 'candidate_index': pl.UInt32, 'item_id': pl.Int32,
                'category_id': pl.Int8, 'timestamp': pl.Int32, 'target': pl.Int8})
            shuffled.write_parquet(prepared / val_entry['candidate_path'])
            shuffled.select('feed_id', 'timestamp').write_parquet(prepared / val_entry['feed_path'])
            val_entry['candidates'] = shuffled.height
            write_json(prepared / 'manifest.json', source)
            ordered_audit = root / 'ordered-audit'
            build_features(prepared, ordered_audit, train_parts=2, smoothing=2., memory_mb=256)
            self.assertTrue(audit(prepared, ordered_audit, root / 'ordered-audit.json')['complete'])


if __name__ == '__main__':
    unittest.main()
