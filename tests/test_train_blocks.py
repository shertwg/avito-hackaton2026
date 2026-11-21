import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import polars as pl

from eval.make_synthetic import make_split
from scripts.bag_train_blocks import bag
from scripts.prepare_train_blocks import prepare_blocks
from scripts.reproduce_data import reproduce
from scripts.run_joint_oof import run
from scripts.run_train_blocks_oof import run_blocks
from src.common.preparation import write_json
from src.common.similarity import build_features as build_cosine
from src.common.infeed import build_features as build_infeed
from src.common.train_selection import select_parts
from src.gbdt.prepared import load_prepared
from src.gbdt.run_exp import get_args, load_all


class TrainSelectionTests(unittest.TestCase):
    def test_legacy_and_contiguous_selection(self):
        source = {'splits': {'train': {'parts': [{'source': str(i)} for i in range(8)]}}}
        self.assertEqual([p['source'] for p in select_parts(source, 3)], ['0', '3', '7'])
        self.assertEqual([p['source'] for p in select_parts(source, 3, 2)], ['2', '3', '4'])
        self.assertEqual([p['source'] for p in select_parts(source, 2, 6)], ['6', '7'])
        for count, start in ((0, None), (9, None), (True, None), (2, -1), (3, 6), (2, True), (2, 1.5)):
            with self.subTest(count=count, start=start), self.assertRaises(ValueError):
                select_parts(source, count, start)
        source['splits']['train']['parts'][1]['source'] = '0'
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            select_parts(source, 2, 0)


class TrainBlockPipelineTests(unittest.TestCase):
    def test_prepare_resume_audits_and_nonzero_block_oof(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, prepared, base = root / 'raw', root / 'prepared', root / 'base'
            rng = np.random.default_rng(42)
            for number, split in enumerate(('train', 'val', 'test')):
                make_split(raw, split, 12, rng, number * 100, split == 'train', split != 'test')
            original = pl.read_parquet(raw / 'train/impressions/part-00000.parquet')
            for number in range(1, 4):
                original.with_columns((pl.col('feed_id') + number * 1000).alias('feed_id')).write_parquet(
                    raw / f'train/impressions/part-{number:05d}.parquet')
            test = pl.read_parquet(raw / 'test/impressions/part-00000.parquet')
            test.select('feed_id', 'item_id').explode('item_id').cast({
                'feed_id': pl.Int32, 'item_id': pl.Int32}).with_columns(
                pl.lit(0., dtype=pl.Float32).alias('score')).write_parquet(raw / 'sample_submission.parquet')
            reproduce(raw, prepared, root / 'eda', root / 'catalogue', base / 'cosine',
                      feature_train_parts=None, infeed=base / 'infeed')
            build_cosine(raw, prepared, root / 'catalogue', base / 'cosine-selected', train_parts=2, train_start=0)
            build_infeed(prepared, base / 'infeed-selected', train_parts=2, train_start=0)
            for name in ('cosine-selected', 'infeed-selected'):
                manifest = json.loads((base / name / 'manifest.json').read_text())
                self.assertEqual(set(manifest['splits']['train']['parts']), {'part-00000.parquet', 'part-00001.parquet'})
            options = dict(prepared=prepared, raw=raw, feature_dir=root / 'blocks', output=root / 'report',
                           starts=[0, 1], train_parts=1, cosine=base / 'cosine-selected', infeed=base / 'infeed-selected',
                           oof_out=root / 'models', weighted=True, memory_mb=256)
            with patch('src.common.click_item2item.BUCKETS', 2):
                click_plan = prepare_blocks(**{**options, 'click_item2item': True, 'starts': [1],
                    'feature_dir': root / 'click-blocks', 'output': root / 'click-report'})
            click_block = click_plan['blocks']['block-001-002']
            self.assertEqual(click_block['model_features'], 139)
            self.assertIn('click-item2item', click_block['audits'])
            plan = prepare_blocks(**options, plan_only=True)
            self.assertEqual(plan['status'], 'planned')
            self.assertFalse((root / 'blocks').exists())
            self.assertFalse((root / 'models').exists())

            def interrupt(path, value):
                write_json(path, value)
                if value.get('stage') == 'block-000-001/itemstats/complete':
                    raise RuntimeError('interrupt after audited itemstats')

            with patch('scripts.prepare_train_blocks.write_json', side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, 'interrupt after'):
                    prepare_blocks(**options, resume=True)
            history = root / 'blocks/block-000-001/history/train/part-00000.parquet'
            stamp = history.stat().st_mtime_ns
            report = prepare_blocks(**options, resume=True)
            self.assertTrue(report['complete'])
            self.assertNotIn('error', report)
            self.assertEqual(history.stat().st_mtime_ns, stamp)
            saved_plan = (root / 'report/run.json').read_bytes()
            prepare_blocks(**options, resume=True)
            self.assertEqual((root / 'report/run.json').read_bytes(), saved_plan)
            for tag, block in report['blocks'].items():
                self.assertTrue(block['complete'])
                self.assertEqual(block['donor_parts'], 3)
                self.assertEqual(block['model_features'], 121)
                self.assertEqual(len(block['audits']), 4)
                for name in ('itemstats', 'item2item', 'weighted-item2item'):
                    manifest = json.loads((root / 'blocks' / tag / name / 'manifest.json').read_text())
                    self.assertEqual(manifest['excluded_train_sources'], block['cohort']['sources'])
                self.assertIn('--train-start', block['oof_argv'])
            self.assertFalse((root / 'models').exists())
            block = report['blocks']['block-001-002']
            loaded, fields = load_prepared(prepared, 'train', block['feature_roots'], train_parts=1, train_start=1)
            self.assertEqual(loaded['feed_id'].min(), 1000)
            self.assertEqual(len(fields), 121)
            args = ['--data', str(raw), '--prepared', str(prepared), '--out', str(root / 'gbdt'),
                    '--train-parts', '1', '--train-start', '1']
            for feature_root in block['feature_roots']:
                args.extend(('--feature-root', feature_root))
            frames, fields = load_all(get_args(args), need_test=True)
            self.assertEqual(frames['train']['feed_id'].min(), 1000)
            self.assertEqual(len(fields), 121)
            out = root / 'models/block-001-002'
            arguments = dict(train_parts=1, train_start=1, folds=4, iterations=2, depth=2, threads=1, memory_mb=256)
            queue = run_blocks(root / 'report/run.json', only=['block-001-002'], iterations=2, depth=2, threads=1, memory_mb=256)
            self.assertTrue(queue['complete'])
            self.assertEqual(queue['requested_blocks'], ['block-001-002'])
            with self.assertRaisesRegex(ValueError, 'all planned blocks'):
                bag(root / 'report/run.json', root / 'bag')
            run_blocks(root / 'report/run.json', only=['block-000-001'], iterations=2, depth=2, threads=1, memory_mb=256)
            bagged = bag(root / 'report/run.json', root / 'bag')
            self.assertEqual(bagged['submission_validator'], 'passed')
            self.assertEqual(bagged['weight_per_block'], .5)
            for split in ('val', 'test'):
                normalized = []
                for tag in report['blocks']:
                    component = pl.read_parquet(root / 'models' / tag / f'{split}_pred.parquet').sort('feed_id', 'item_id')
                    s = pl.col('score').cast(pl.Float64)
                    normalized.append(component.select(
                        ((s - s.mean().over('feed_id')) / (s.std().over('feed_id') + 1e-9)).fill_null(0))[:, 0].to_numpy())
                actual = pl.read_parquet(root / 'bag' / f'{split}_pred.parquet').sort('feed_id', 'item_id')
                np.testing.assert_allclose(actual['score'].to_numpy(), np.mean(normalized, axis=0), atol=1e-6)
            result = json.loads((out / 'run.json').read_text())
            self.assertTrue(result['complete'])
            self.assertEqual(result['identity']['train_sources'], ['part-00001.parquet'])
            self.assertEqual(len(result['identity']['features']), 121)
            with self.assertRaisesRegex(ValueError, 'does not cover'):
                run(prepared, raw, root / 'wrong-oof', block['feature_roots'], **{**arguments, 'train_start': 0})

            result['identity']['implementation_sha256'] = 'archived-runner-version'
            result['identity']['code_sha256']['src/common/similarity.py'] = 'archived-dependency-version'
            write_json(out / 'run.json', result)
            with patch('scripts.run_joint_oof.export_training', side_effect=AssertionError('completed run rebuilt')):
                restored = run(prepared, raw, out, block['feature_roots'], resume=True, **arguments)
            self.assertEqual(restored['identity']['implementation_sha256'], 'archived-runner-version')
            self.assertEqual(restored['identity']['code_sha256']['src/common/similarity.py'], 'archived-dependency-version')
            result['complete'] = False
            write_json(out / 'run.json', result)
            with self.assertRaisesRegex(ValueError, 'matching identity'):
                run(prepared, raw, out, block['feature_roots'], resume=True, **arguments)
            for starts, count in (([0, 0], 1), ([0, 1], 2), ([3], 2)):
                with self.subTest(starts=starts), self.assertRaises(ValueError):
                    prepare_blocks(**{**options, 'starts': starts, 'train_parts': count,
                                      'output': root / 'invalid'}, plan_only=True)
            manifest = root / 'blocks/block-001-002/itemstats/manifest.json'
            itemstats = json.loads(manifest.read_text())
            itemstats['excluded_train_sources'] = ['part-00000.parquet']
            write_json(manifest, itemstats)
            with self.assertRaisesRegex(ValueError, 'audit changed'):
                prepare_blocks(**options, resume=True)


if __name__ == '__main__':
    unittest.main()
