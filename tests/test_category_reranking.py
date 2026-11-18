import unittest

import numpy as np
import polars as pl

from scripts.probe_category_reranking import category_rank, summarize_late_feeds


class CategoryRerankingTests(unittest.TestCase):
    def test_ties_singletons_and_feed_boundaries(self):
        frame = pl.DataFrame({'feed_id': [2, 1, 1, 1, 1], 'category_id': [8, 8, 8, 8, 9],
                              'f1': [99, 2, 5, 2, 3]})
        np.testing.assert_allclose(frame.select(category_rank()).to_series(), [.5, 1/3, 5/6, 1/3, .5])
        np.testing.assert_allclose(frame.reverse().select(category_rank()).to_series().reverse(),
                                   frame.select(category_rank()).to_series())

    def test_slice_delta_is_joined_by_feed_id(self):
        frame = pl.DataFrame({'feed_id': [2, 1, 2], 'category_id': [8, 8, 9]})
        result = summarize_late_feeds(frame, np.array([1, 2]), np.array([.3, -.1]))
        self.assertEqual(result['category_count']['1']['improved'], 1)
        self.assertEqual(result['category_count']['2-5']['mean_delta'], -.1)
