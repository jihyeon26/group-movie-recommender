"""Tests for the disjoint-pair robustness experiment."""

from __future__ import annotations

import unittest

import pandas as pd

from group_movie_recommender.pipelines.scaled_pair_experiment import (
    _validate_settings,
    select_disjoint_pairs,
)


class ScaledPairExperimentTests(unittest.TestCase):
    def test_disjoint_selection_is_deterministic_and_has_no_repeated_users(self) -> None:
        pairs = pd.DataFrame({
            "userA": [1, 1, 3, 5, 7],
            "userB": [2, 3, 4, 6, 8],
            "genreSimilarity": [-0.1, -0.2, -0.3, -0.4, -0.5],
        })
        first = select_disjoint_pairs(pairs, pair_limit=3)
        second = select_disjoint_pairs(pairs, pair_limit=3)
        members = pd.concat([first["userA"], first["userB"]])

        pd.testing.assert_frame_equal(first, second)
        self.assertFalse(members.duplicated().any())
        self.assertEqual(first["pairId"].tolist(), [0, 1, 2])

    def test_nested_counts_must_end_at_pair_limit(self) -> None:
        settings = {
            "subset": {
                "pair_limit": 500,
                "nested_pair_counts": [100, 250],
                "total_user_limit": 5000,
            }
        }
        with self.assertRaises(ValueError):
            _validate_settings(settings)


if __name__ == "__main__":
    unittest.main()
