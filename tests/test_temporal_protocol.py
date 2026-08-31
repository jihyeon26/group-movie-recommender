"""Tests for period-specific history and relevance construction."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from group_movie_recommender.evaluation.temporal_protocol import build_temporal_context
from group_movie_recommender.preprocessing.config import PreprocessingConfig


class TemporalProtocolTests(unittest.TestCase):
    def test_test_history_includes_validation_and_removes_partner_seen_items(self) -> None:
        config = PreprocessingConfig(train_end="2019-01-01", validation_end="2020-01-01")
        pairs = pd.DataFrame({"pairId": [0], "userA": [1], "userB": [2]})
        train = config.train_end_timestamp - 1
        validation = config.train_end_timestamp + 1
        test = config.validation_end_timestamp + 1
        ratings = pd.DataFrame(
            {
                "userId": [1, 1, 1, 2, 2, 2],
                "movieId": [100, 200, 300, 200, 300, 400],
                "rating": [5.0] * 6,
                "timestamp": [train, validation, test, train, validation, test],
            }
        )

        context = build_temporal_context(
            pairs, ratings, np.array([100, 200, 300, 400]), config, period="test"
        )

        self.assertEqual(context.seen[1], {100, 200})
        self.assertEqual(context.seen[2], {200, 300})
        self.assertEqual(context.relevance[(0, 1)], set())
        self.assertEqual(context.relevance[(0, 2)], {400})
        self.assertEqual(context.diagnostics["history_splits"], ["train", "validation"])


if __name__ == "__main__":
    unittest.main()
