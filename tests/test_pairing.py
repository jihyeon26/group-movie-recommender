"""Tests for deterministic dissimilar-pair selection."""

from __future__ import annotations

import unittest

import pandas as pd

from group_movie_recommender.config import PreprocessingConfig
from group_movie_recommender.pairing import select_dissimilar_pairs


class PairingTests(unittest.TestCase):
    def test_pair_selection_applies_thresholds_and_user_cap(self) -> None:
        features = pd.DataFrame(
            {
                "userA": [1, 1, 1, 2, 3, 4],
                "userB": [2, 3, 4, 3, 4, 5],
                "coRatedTrain": [30, 30, 30, 30, 30, 30],
                "genreSimilarity": [-0.8, -0.7, -0.6, 0.1, 0.2, 0.3],
                "ratingCorrelation": [-0.7, -0.6, -0.5, 0.1, 0.2, 0.3],
            }
        )
        config = PreprocessingConfig(
            pair_min_co_rated=20,
            pair_similarity_quantile=0.5,
            max_pairs_per_user=1,
            random_seed=7,
        )
        selected, summary = select_dissimilar_pairs(features, config)

        used_users = pd.concat([selected["userA"], selected["userB"]]).value_counts()
        self.assertTrue((used_users <= 1).all())
        self.assertGreaterEqual(summary["dissimilar_candidates"], len(selected))


if __name__ == "__main__":
    unittest.main()
