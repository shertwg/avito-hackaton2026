import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import polars as pl

from scripts.audit_item2item_features import audit
from src.common.item2item import ItemHistoryIndex, build_feature_part, build_features, checkpoint_table
from src.common.preparation import fingerprint, write_json
from src.common.similarity import attach_features


class Item2ItemTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.prepared = self.root / 'prepared'
        histories = {'train': [[[1, 2]], [[1, 1, 2, 4, 5], []], [[1, 3], [2]], [[2] * 10 + [3] * 40 + [1]]],
                     'val': [[[2, 3]]], 'test': [[[]]]}
        source = {'complete': True, 'splits': {}}
        for split, parts in histories.items():
            entries = []
            for index, history in enumerate(parts):
                ids = [index * 10 + offset for offset in range(len(history))]
                feeds = pl.DataFrame({'feed_id': ids, 'history__item_id': history}, schema_overrides={
                    'feed_id': pl.Int32, 'history__item_id': pl.List(pl.Int32)})
                rows = [{'split': split, 'feed_id': feed, 'candidate_index': position, 'item_id': item,
                         'target': position % 2} for feed in ids for position, item in enumerate((1, 1, 4, 99, 0, -1))]
                candidates = pl.DataFrame(rows, schema_overrides={
                    'feed_id': pl.Int32, 'candidate_index': pl.UInt32, 'item_id': pl.Int32, 'target': pl.Int8})
                name = f'part-{index:05d}.parquet'
                paths = {'candidate_path': f'{split}/candidates/{name}', 'feed_path': f'{split}/feeds/{name}'}
                for key, frame in (('candidate_path', candidates), ('feed_path', feeds)):
                    path = self.prepared / paths[key]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    frame.write_parquet(path)
                entries.append({'source': name, **paths, 'candidates': candidates.height})
            source['splits'][split] = {'parts': entries}
        write_json(self.prepared / 'manifest.json', source)

    def build(self, name, resume=False):
        return build_features(self.prepared, self.root / name, train_parts=2, memory_mb=256, resume=resume)

    def test_exact_pairs_popularity_windows_and_label_exclusion(self):
        manifest = self.build('features')
        output = self.root / 'features'
        first = pl.read_parquet(output / 'train/part-00000.parquet')
        np.testing.assert_allclose(first['i2i_sum_10'], [1.5, 1.5, np.sqrt(2), 0., 0., 0.])
        np.testing.assert_allclose(first['i2i_max_10'], [1., 1., 1 / np.sqrt(2), 0., 0., 0.])
        long = pl.read_parquet(output / 'train/part-00003.parquet')
        self.assertAlmostEqual(long['i2i_sum_10'][0], 5.)
        self.assertAlmostEqual(long['i2i_sum_50'][0], 5 + 40 / np.sqrt(2), places=5)
        self.assertAlmostEqual(long['i2i_max_50'][0], 1 / np.sqrt(2), places=6)
        with ItemHistoryIndex(output) as index:
            self.assertEqual(index.metadata['postings'], 6)
            self.assertEqual(index.items.tolist(), [1, 2, 3, 4])
            self.assertEqual(index.offsets.tolist(), [0, 2, 4, 5, 6])
            candidates = pl.read_parquet(self.prepared / 'train/candidates/part-00003.parquet')
            feeds = pl.read_parquet(self.prepared / 'train/feeds/part-00003.parquet')
            self.assertTrue(build_feature_part(candidates, feeds, index, candidate_batch=1).equals(long))
        self.assertTrue(audit(self.prepared, output, self.root / 'audit.json')['complete'])
        with self.assertRaisesRegex(ValueError, 'donor shards overlap'):
            attach_features(first, output, 'train', ['part-00001.parquet'])
        for path in self.prepared.glob('*/candidates/*.parquet'):
            pl.read_parquet(path).with_columns((1 - pl.col('target')).alias('target')).write_parquet(path)
        self.build('changed')
        for path in output.glob('*/*.parquet'):
            self.assertTrue(pl.read_parquet(path).equals(pl.read_parquet(self.root / 'changed' / path.relative_to(output))))
        changed = self.root / 'changed'
        postings = np.load(changed / '_feeds.npy', mmap_mode='r+')
        postings[1] = 22
        postings.flush()
        del postings
        metadata = json.loads((changed / '_index.json').read_text())
        metadata['files']['_feeds.npy'] = fingerprint(changed / '_feeds.npy')
        write_json(changed / '_index.json', metadata)
        saved = json.loads((changed / 'manifest.json').read_text())
        saved['artifacts']['_feeds.npy'] = fingerprint(changed / '_feeds.npy')
        saved['artifacts']['_index.json'] = fingerprint(changed / '_index.json')
        write_json(changed / 'manifest.json', saved)
        with self.assertRaisesRegex(ValueError, 'independent item posting mismatch'):
            audit(self.prepared, changed, self.root / 'invalid.json')
        self.assertEqual(len(manifest['excluded_train_sources']), 2)

    def test_table_and_feature_checkpoints_resume(self):
        def interrupt_table(connection, name, query):
            checkpoint_table(connection, name, query)
            if name == 'query_items':
                raise RuntimeError('table interruption')
        with patch('src.common.item2item.checkpoint_table', side_effect=interrupt_table):
            with self.assertRaisesRegex(RuntimeError, 'table interruption'):
                self.build('resume')
        def interrupt_feature(path, value):
            write_json(path, value)
            if value.get('stage') == 'features_train' and len(value['splits']['train']['parts']) == 1:
                raise RuntimeError('feature interruption')
        with patch('src.common.feature_cache.write_json', side_effect=interrupt_feature):
            with self.assertRaisesRegex(RuntimeError, 'feature interruption'):
                self.build('resume', resume=True)
        path = self.root / 'resume/train/part-00000.parquet'
        before = path.stat().st_mtime_ns
        with patch('src.common.item2item.checkpoint_table', side_effect=AssertionError('aggregate should be reused')):
            self.build('resume', resume=True)
        self.assertEqual(path.stat().st_mtime_ns, before)
        self.assertTrue(audit(self.prepared, self.root / 'resume', self.root / 'resume-audit.json')['complete'])

    def test_interrupted_multi_array_publication_recovers(self):
        def interrupt_publish(output):
            (output / '_items.npy.tmp').replace(output / '_items.npy')
            raise RuntimeError('array interruption')
        with patch('src.common.item2item.publish_arrays', side_effect=interrupt_publish):
            with self.assertRaisesRegex(RuntimeError, 'array interruption'):
                self.build('arrays')
        self.assertFalse((self.root / 'arrays/_index.json').exists())
        self.build('arrays', resume=True)
        self.assertTrue(audit(self.prepared, self.root / 'arrays', self.root / 'array-audit.json')['complete'])

    def test_empty_donor_index(self):
        for path in self.prepared.glob('train/feeds/*.parquet'):
            frame = pl.read_parquet(path)
            frame.with_columns(pl.Series('history__item_id', [[]] * frame.height, dtype=pl.List(pl.Int32))).write_parquet(path)
        self.build('empty')
        with ItemHistoryIndex(self.root / 'empty') as index:
            self.assertEqual(index.metadata['postings'], 0)
            self.assertEqual(index.items.tolist(), [])
        for path in (self.root / 'empty').glob('*/*.parquet'):
            frame = pl.read_parquet(path)
            self.assertTrue((frame.select('i2i_sum_10', 'i2i_max_50').to_numpy() == 0).all())
        self.assertTrue(audit(self.prepared, self.root / 'empty', self.root / 'empty-audit.json')['complete'])


if __name__ == '__main__':
    unittest.main()
