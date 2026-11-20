from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import polars as pl

from scripts.compare_team_overlay import compare


class TeamOverlayTests(unittest.TestCase):
    def test_selection_uses_only_public_and_rejects_bad_keys(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            base, extra = root / 'base', root / 'extra'
            base.mkdir()
            extra.mkdir()
            labels = pl.DataFrame({'feed_id': [i for i in range(10) for _ in range(2)],
                                   'item_id': [1, 2] * 10, 'target': [1, 0] * 10},
                                  schema={'feed_id': pl.Int32, 'item_id': pl.Int32, 'target': pl.Int8})
            labels = labels.with_columns(pl.col('feed_id').alias('timestamp'))
            for path, direction in [(base, -1), (extra, 1)]:
                signed = pl.when(pl.col('feed_id') < 3).then(direction).otherwise(-direction)
                frame = labels.with_columns((signed * pl.col('target')).cast(pl.Float32).alias('score'))
                frame.drop('timestamp').reverse().write_parquet(path / 'val_pred.parquet')
                frame.select('feed_id', 'item_id', 'score').write_parquet(path / 'test_pred.parquet')
            frame.select('feed_id', 'item_id', 'score').write_parquet(root / 'sample_submission.parquet')
            with patch('scripts.compare_team_overlay.load_labels', return_value=labels):
                result = compare(root, base, extra, root / 'out')
                self.assertGreater(result['candidate_weight'], .5)
                self.assertLess(result['private_delta'], 0)
                self.assertEqual(result['submission_validator'], 'passed')
                bad = pl.read_parquet(extra / 'test_pred.parquet')
                pl.concat([bad.head(1), bad.head(bad.height-1)]).write_parquet(extra / 'test_pred.parquet')
                with self.assertRaisesRegex(ValueError, 'keys differ'):
                    compare(root, base, extra, root / 'bad')
                self.assertFalse((root / 'bad').exists())
