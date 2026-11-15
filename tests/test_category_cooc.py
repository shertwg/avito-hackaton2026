import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import polars as pl

from scripts.audit_category_cooc_features import audit
from src.common.category_cooc import build_features, history_presence, save_counts
from src.common.preparation import fingerprint, write_json
from src.common.similarity import attach_features


class CategoryCooccurrenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.prepared = self.root / 'prepared'
        histories = {'train': [[[1, 2]], [[1, 1, 2], []], [[1, 3], [2]], [[2] * 10 + [3] * 40]],
                     'val': [[[2, 3]]], 'test': [[[]]]}
        source = {'complete': True, 'splits': {}}
        for split, parts in histories.items():
            entries = []
            for index, values in enumerate(parts):
                feed_ids = [index * 10 + offset for offset in range(len(values))]
                feeds = pl.DataFrame({'feed_id': feed_ids, 'history__category_id': values},
                                     schema_overrides={'feed_id': pl.Int32, 'history__category_id': pl.List(pl.Int8)})
                records = [{'split': split, 'feed_id': feed, 'candidate_index': position,
                            'item_id': 10 if position < 2 else 99, 'category_id': category,
                            'target': position % 2}
                           for feed in feed_ids for position, category in enumerate((1, 1, 99))]
                candidates = pl.DataFrame(records, schema_overrides={
                    'feed_id': pl.Int32, 'candidate_index': pl.UInt32, 'item_id': pl.Int32,
                    'category_id': pl.Int8, 'target': pl.Int8})
                name = f'part-{index:05d}.parquet'
                paths = {'candidate_path': f'{split}/candidates/{name}', 'feed_path': f'{split}/feeds/{name}'}
                for key, frame in (('candidate_path', candidates), ('feed_path', feeds)):
                    path = self.prepared / paths[key]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    frame.write_parquet(path)
                entries.append({'source': name, **paths, 'candidates': candidates.height})
            source['splits'][split] = {'parts': entries}
        write_json(self.prepared / 'manifest.json', source)

    def test_manual_formulas_audit_and_label_exclusion(self):
        cache = self.root / 'cache'
        manifest = build_features(self.prepared, cache, train_parts=2)
        frame = pl.read_parquet(cache / 'train/part-00000.parquet')
        self.assertEqual(frame.height, 3)
        np.testing.assert_allclose(frame['cc_sum_10'], [1.5, 1.5, 0.])
        np.testing.assert_allclose(frame['cc_max_10'], [1., 1., 0.])
        long = pl.read_parquet(cache / 'train/part-00003.parquet')
        self.assertAlmostEqual(long['cc_sum_10'][0], 5.)
        self.assertAlmostEqual(long['cc_sum_50'][0], 5 + 40 / np.sqrt(2), places=5)
        self.assertAlmostEqual(long['cc_max_50'][0], 1 / np.sqrt(2), places=6)
        self.assertTrue(audit(self.prepared, cache, self.root / 'audit.json')['complete'])
        for path in self.prepared.glob('*/candidates/*.parquet'):
            pl.read_parquet(path).with_columns((1 - pl.col('target')).alias('target')).write_parquet(path)
        changed = self.root / 'changed'
        build_features(self.prepared, changed, train_parts=2)
        for path in cache.glob('*/*.parquet'):
            self.assertTrue(pl.read_parquet(path).equals(pl.read_parquet(changed / path.relative_to(cache))))
        with self.assertRaisesRegex(ValueError, 'donor shards overlap'):
            attach_features(frame, cache, 'train', ['part-00001.parquet'])
        # A finite, correctly fingerprinted wrong value must fail the independent formulas.
        corrupt = changed / 'val/part-00000.parquet'
        pl.read_parquet(corrupt).with_columns((pl.col('cc_sum_10') + 0.1).alias('cc_sum_10')).write_parquet(corrupt)
        saved = json.loads((changed / 'manifest.json').read_text())
        saved['splits']['val']['parts']['part-00000.parquet']['fingerprint'] = fingerprint(corrupt)
        write_json(changed / 'manifest.json', saved)
        with self.assertRaisesRegex(ValueError, 'formula mismatch'):
            audit(self.prepared, changed, self.root / 'invalid.json')
        self.assertEqual(manifest['donors_processed'], 2)
        presence, categories = history_presence(pl.DataFrame({'history__category_id': [[-128, 127]]},
                                                schema_overrides={'history__category_id': pl.List(pl.Int8)}))
        self.assertEqual(categories.tolist(), [0, 255])
        self.assertEqual(presence.tolist(), [[1., 1.]])

    def test_checkpoint_resume_preserves_completed_parts(self):
        cache = self.root / 'cache'
        def interrupted(path, counts, processed, identity):
            save_counts(path, counts, processed, identity)
            if processed == 1:
                raise RuntimeError('donor interruption')
        with patch('src.common.category_cooc.save_counts', side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, 'donor interruption'):
                build_features(self.prepared, cache, train_parts=2)
        def feature_interruption(path, value):
            write_json(path, value)
            if value.get('stage') == 'features_train' and len(value['splits']['train']['parts']) == 1:
                raise RuntimeError('feature interruption')
        with patch('src.common.category_cooc.write_json', side_effect=feature_interruption):
            with self.assertRaisesRegex(RuntimeError, 'feature interruption'):
                build_features(self.prepared, cache, train_parts=2, resume=True)
        first = cache / 'train/part-00000.parquet'
        before = first.stat().st_mtime_ns
        build_features(self.prepared, cache, train_parts=2, resume=True)
        self.assertEqual(first.stat().st_mtime_ns, before)
        self.assertTrue(audit(self.prepared, cache, self.root / 'resumed.json')['complete'])
        with self.assertRaisesRegex(ValueError, 'sources changed'):
            build_features(self.prepared, cache, train_parts=1, resume=True)


if __name__ == '__main__':
    unittest.main()
