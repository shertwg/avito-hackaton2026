from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import polars as pl

from eval.make_synthetic import make_split
from scripts.reproduce_ranking import reproduce_ranking
from scripts.run_joint_oof import export_training, run
from scripts.validate_submission import validate
from src.common.preparation import write_json


class JointOofTests(unittest.TestCase):
    def test_disk_oof_exclusion_and_fold_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, prepared, features = root / 'raw', root / 'prepared', root / 'features'
            rng = np.random.default_rng(42)
            for number, split in enumerate(('train', 'val', 'test')):
                make_split(raw, split, 12, rng, number * 100, split == 'train', split != 'test')
            donor = pl.read_parquet(raw / 'train/impressions/part-00000.parquet').with_columns(
                (pl.col('feed_id') + 1000).alias('feed_id'))
            donor.write_parquet(raw / 'train/impressions/part-00001.parquet')
            test = pl.read_parquet(raw / 'test/impressions/part-00000.parquet')
            test.select('feed_id', 'item_id').explode('item_id').cast({'feed_id': pl.Int32, 'item_id': pl.Int32}).with_columns(
                pl.lit(0., dtype=pl.Float32).alias('score')).write_parquet(raw / 'sample_submission.parquet')
            reproduce_ranking(raw, root / 'prepare-check', prepared=prepared, feature_dir=features,
                catalogue=root / 'catalogue', train_parts=1, infeed=True, history=True, itemstats=True,
                item2item=True, iterations=2, depth=2, threads=1, memory_mb=256)
            roots = [features / name for name in ('cosine', 'infeed', 'history', 'itemstats-001', 'item2item-001')]
            arguments = dict(train_parts=1, folds=4, iterations=2, depth=2, threads=1, memory_mb=256)
            out = root / 'oof'
            observed = []

            def checked_export(paths, assignments, held, names, target, **options):
                export_training(paths, assignments, held, names, target, **options)
                rows = pl.read_csv(target, separator='\t', has_header=False, columns=[1])
                held_ids = set(assignments.filter(pl.col('fold') == held)['feed_id'].to_list())
                self.assertFalse(set(rows.to_series().to_list()) & held_ids)
                observed.append(held)

            def interrupt(path, value):
                write_json(path, value)
                if value.get('stage') == 'fold_0_complete':
                    raise RuntimeError('interrupt after first fold')

            with patch('scripts.run_joint_oof.export_training', side_effect=checked_export), \
                    patch('scripts.run_joint_oof.write_json', side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, 'interrupt after first fold'):
                    run(prepared, raw, out, roots, **arguments)
            first_model_time = (out / 'fold-0/model.cbm').stat().st_mtime_ns
            with patch('scripts.run_joint_oof.export_training', side_effect=checked_export):
                report = run(prepared, raw, out, roots, resume=True, **arguments)
            self.assertTrue(report['complete'])
            self.assertEqual(observed, [0, 1, 2, 3])
            self.assertEqual((out / 'fold-0/model.cbm').stat().st_mtime_ns, first_model_time)
            self.assertEqual(len(report['identity']['features']), 109)
            self.assertEqual(len(report['folds']), 4)
            self.assertFalse(validate(str(raw / 'sample_submission.parquet'), str(out / 'test_pred.parquet')))
            self.assertTrue(all(v['reference_check'] == 'passed' for v in report['validation'].values()))
            before = (out / 'val_pred.parquet').stat().st_mtime_ns
            with patch('scripts.run_joint_oof.export_training', side_effect=AssertionError('completed run rebuilt')):
                run(prepared, raw, out, roots, resume=True, **arguments)
            self.assertEqual((out / 'val_pred.parquet').stat().st_mtime_ns, before)
            (out / 'val_pred.parquet').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                run(prepared, raw, out, roots, resume=True, **arguments)


if __name__ == '__main__':
    unittest.main()
