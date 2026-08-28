"""Tests for temporal splitting and user cohort selection."""

from __future__ import annotations

from datetime import datetime, timezone
import unittest

import pandas as pd

from group_movie_recommender.config import PreprocessingConfig
from group_movie_recommender.splitting import (
    add_temporal_split,
    select_evaluation_users,
    select_training_users,
    user_split_statistics,
)


def utc_epoch(date_text: str) -> int:
    return int(datetime.strptime(date_text, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


class SplittingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = PreprocessingConfig(
            training_min_positives=1,
            evaluation_min_train_ratings=2,
            evaluation_min_train_positives=1,
            evaluation_min_validation_positives=1,
            evaluation_min_test_positives=1,
        )
        self.ratings = pd.DataFrame(
            {
                "userId": [1, 1, 1, 1, 2, 2],
                "movieId": [1, 2, 3, 4, 1, 5],
                "rating": [4.0, 3.0, 4.5, 5.0, 4.0, 4.0],
                "timestamp": [
                    utc_epoch("2018-01-01"),
                    utc_epoch("2018-02-01"),
                    utc_epoch("2019-03-01"),
                    utc_epoch("2020-04-01"),
                    utc_epoch("2018-01-01"),
                    utc_epoch("2020-01-01"),
                ],
            }
        )

    def test_split_and_cohorts(self) -> None:
        split = add_temporal_split(self.ratings, self.config)
        self.assertEqual(split["split"].astype(str).tolist()[:4], ["train", "train", "validation", "test"])

        statistics = user_split_statistics(split)
        self.assertEqual(select_training_users(statistics, self.config).tolist(), [1, 2])
        self.assertEqual(select_evaluation_users(statistics, self.config).tolist(), [1])


if __name__ == "__main__":
    unittest.main()

