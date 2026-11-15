from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import polars as pl

from scripts.audit_event_cosine_features import audit as audit_event
from scripts.audit_feature_ranks import audit as audit_ranks
from scripts.audit_time_features import audit as audit_time
from src.common.event_cosine import build_feature_part, build_features as build_event
from src.common.feature_cache import KEY, cache_lock
from src.common.feature_ranks import build_features as build_ranks
from src.common.preparation import fingerprint, write_json
from src.common.similarity import attach_features
from src.common.time_features import add_time_fields, build_features as build_time
from tests.new_feature_fixture import make_prepared, rank_sources


class AdditionalFeatureTests(unittest.TestCase):
    def test_event_masks_negative_max_and_original_window(self):
        vectors = np.zeros((4, 128), dtype=np.int8)
        vectors[0, 0], vectors[1, 0], vectors[2, 1] = 1, -1, 1
        candidates = pl.DataFrame({'split': ['val'] * 5, 'feed_id': [1, 1, 1, 1, 2],
            'candidate_index': [0, 1, 2, 3, 0], 'item_id': [7, 7, 8, 9, 10],
            'item_id__embedding_idx': [0, 1, -1, 3, 0]}, schema_overrides={
                'feed_id': pl.Int32, 'candidate_index': pl.UInt32, 'item_id': pl.Int32,
                'item_id__embedding_idx': pl.Int32})
        feeds = pl.DataFrame({'feed_id': [1, 2],
            'history__item_id__embedding_idx': [[1, -1, 2, 0], [-1] * 50 + [0]],
            'history__event_type': [[10, 10, 1, 2], [1] * 50 + [10]]}, schema_overrides={
                'feed_id': pl.Int32, 'history__item_id__embedding_idx': pl.List(pl.Int32),
                'history__event_type': pl.List(pl.Int8)})
        frame = build_feature_part(candidates, feeds, vectors, np.arange(4))
        np.testing.assert_allclose(frame['cos_evt10_max_50'], [-1, 1, 0, 0, 0])
        np.testing.assert_allclose(frame['hist_evt10_valid_50'], [1, 1, 1, 1, 0])
        np.testing.assert_allclose(frame['cos_evt2_mean_50'], [1, -1, 0, 0, 0])
        self.assertEqual(frame.height, 5)
        self.assertTrue(frame.select(KEY).equals(candidates.select(KEY)))

    def test_event_cache_resume_expansion_audit_and_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, prepared, source = make_prepared(root)
            output = root / 'event'
            def interrupt(path, value):
                write_json(path, value)
                if value.get('stage') == 'features_train' and len(value['splits']['train']['parts']) == 1:
                    raise RuntimeError('part interruption')
            with patch('src.common.feature_cache.write_json', side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, 'part interruption'):
                    build_event(raw, prepared, output, train_parts=3)
            part = output / 'train' / source['splits']['train']['parts'][0]['source']
            timestamp = part.stat().st_mtime_ns
            build_event(raw, prepared, output, train_parts=3, resume=True, prefetch_parts=1)
            self.assertEqual(part.stat().st_mtime_ns, timestamp)
            manifest = build_event(raw, prepared, output, train_parts=2, resume=True)
            self.assertEqual(manifest['splits']['train']['covered_parts'], 3)
            self.assertTrue(audit_event(raw, prepared, output, root / 'event-audit.json', train_parts=2)['complete'])
            with cache_lock(root / 'locked'):
                with self.assertRaisesRegex(ValueError, 'another writer'):
                    build_event(raw, prepared, root / 'locked', train_parts=2)
            vector = raw / 'val/item_embeddings/data.memmap'
            values = np.fromfile(vector, dtype=np.int8)
            values[0] = -values[0] if values[0] else 1
            values.tofile(vector)
            with self.assertRaisesRegex(ValueError, 'sources or implementation changed'):
                build_event(raw, prepared, output, train_parts=2, resume=True)

    def test_rank_ties_dependency_cohort_and_independent_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, prepared, source = make_prepared(root)
            cosine, itemstats = rank_sources(prepared, source, root)
            output = root / 'ranks'
            manifest = build_ranks(prepared, cosine, itemstats, output, train_parts=2)
            self.assertTrue(audit_ranks(prepared, cosine, itemstats, output, root / 'rank-audit.json')['complete'])
            part = manifest['splits']['val']['parts'][source['splits']['val']['parts'][0]['source']]
            path = output / part['path']
            frame = pl.read_parquet(path)
            before = path.stat().st_mtime_ns
            dependency = json.loads((cosine / 'manifest.json').read_text())
            dependency['duration_seconds'] = 1234.
            write_json(cosine / 'manifest.json', dependency)
            build_ranks(prepared, cosine, itemstats, output, train_parts=2, resume=True)
            self.assertEqual(path.stat().st_mtime_ns, before)
            for feed in frame['feed_id'].unique():
                values = frame.filter(pl.col('feed_id') == feed)
                expected = (values.height + 1) / (2 * values.height)
                np.testing.assert_allclose(values['fr_cos_hist_max_50'], expected)
            with self.assertRaisesRegex(ValueError, 'exclusion cohort differs'):
                build_ranks(prepared, cosine, itemstats, root / 'wrong', train_parts=1)
            with self.assertRaisesRegex(ValueError, 'donor shards overlap'):
                attach_features(frame, output, 'train', [source['splits']['train']['parts'][1]['source']])
            changed = frame.with_columns((pl.col('fr_cos_hist_max_50') / 2).alias('fr_cos_hist_max_50'))
            changed.write_parquet(path)
            manifest['splits']['val']['parts'][source['splits']['val']['parts'][0]['source']]['fingerprint'] = fingerprint(path)
            write_json(output / 'manifest.json', manifest)
            with self.assertRaisesRegex(ValueError, 'formula mismatch'):
                audit_ranks(prepared, cosine, itemstats, output, root / 'invalid-rank.json')

    def test_time_strict_past_and_utc_calendar(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, prepared, source = make_prepared(root)
            for split, info in source['splits'].items():
                for index, entry in enumerate(info['parts']):
                    path = prepared / entry['candidate_path']
                    frame = pl.read_parquet(path)
                    items = np.resize([10, 20, 30, 0], frame.height)
                    if split == 'train' and index in (1, 2):
                        first_feed = frame['feed_id'][0]
                        old = frame['feed_id'].to_numpy() == first_feed
                        items = np.where(old, 10, 20 if index == 1 else 30)
                        timestamps = np.where(old, 90 if index == 1 else 95, 200 if index == 1 else 100)
                    else:
                        timestamps = [300 if split == 'test' or (split == 'train' and index == 3) else 100] * frame.height
                    frame.with_columns(pl.Series('item_id', items, dtype=pl.Int32),
                                       pl.Series('timestamp', timestamps, dtype=pl.Int32)).write_parquet(path)
            output = root / 'time'
            manifest = build_time(prepared, output, train_parts=2, memory_mb=256)
            self.assertTrue(audit_time(prepared, output, root / 'time-audit.json')['complete'])
            entry = source['splits']['val']['parts'][0]
            frame = pl.read_parquet(output / manifest['splits']['val']['parts'][entry['source']]['path'])
            self.assertEqual(frame['item_seen_before'].head(4).to_list(), [1., 0., 0., 0.])
            self.assertAlmostEqual(frame['item_train_age_days'][0], 10 / 86400., places=9)
            self.assertEqual(frame['time_weekday_utc'][0], 3.)
            first = (output / manifest['splits']['train']['parts'][source['splits']['train']['parts'][0]['source']]['path'])
            timestamp = first.stat().st_mtime_ns
            build_time(prepared, output, train_parts=2, memory_mb=256, resume=True)
            self.assertEqual(first.stat().st_mtime_ns, timestamp)
            clock = pl.DataFrame({'split': ['val'], 'feed_id': [0], 'candidate_index': [0], 'item_id': [0],
                                   'timestamp': [-1], 'first_seen': [None]}, schema_overrides={
                                       'feed_id': pl.Int32, 'candidate_index': pl.UInt32,
                                       'item_id': pl.Int32, 'first_seen': pl.Int64})
            value = add_time_fields(clock)
            self.assertEqual(value['time_hour_utc'][0], 23.)
            self.assertEqual(value['time_weekday_utc'][0], 2.)


if __name__ == '__main__':
    unittest.main()
import json
