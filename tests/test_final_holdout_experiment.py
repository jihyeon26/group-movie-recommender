"""Tests for pre-test eligibility in the final holdout protocol."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from group_movie_recommender.pipelines.final_holdout_experiment import (
    assign_similarity_bands,
    pretest_eligible_user_ids,
    random_disjoint_pairs,
    similarity_band_rule,
)


class FinalHoldoutExperimentTests(unittest.TestCase):
    def test_eligibility_uses_only_declared_pretest_columns(self) -> None:
        statistics = pd.DataFrame({
            "userId": [1, 2, 3, 4],
            "train_interactions": [100, 99, 100, 100],
            "train_positives": [20, 20, 19, 20],
            "validation_positives": [3, 3, 3, 2],
            "test_positives": [0, 100, 100, 100],
        })
        result = pretest_eligible_user_ids(
            statistics,
            min_train_ratings=100,
            min_train_positives=20,
            min_validation_positives=3,
            excluded_user_ids=np.array([], dtype=np.int64),
        )

        np.testing.assert_array_equal(result, np.array([1]))

    def test_prior_users_are_excluded(self) -> None:
        statistics = pd.DataFrame({
            "userId": [1, 2],
            "train_interactions": [100, 100],
            "train_positives": [20, 20],
            "validation_positives": [3, 3],
        })
        result = pretest_eligible_user_ids(
            statistics,
            min_train_ratings=100,
            min_train_positives=20,
            min_validation_positives=3,
            excluded_user_ids=np.array([1]),
        )

        np.testing.assert_array_equal(result, np.array([2]))

    def test_random_pairs_are_reproducible_and_member_disjoint(self) -> None:
        first = random_disjoint_pairs(np.arange(20), pair_count=5, random_seed=7)
        second = random_disjoint_pairs(np.arange(20), pair_count=5, random_seed=7)
        members = pd.concat([first["userA"], first["userB"]])

        pd.testing.assert_frame_equal(first, second)
        self.assertFalse(members.duplicated().any())

    def test_similarity_bands_are_frozen_from_development_pairs(self) -> None:
        development = pd.DataFrame({"genreSimilarity": [-0.5, 0.0, 0.2, 0.4, 0.8]})
        rule = similarity_band_rule(development)
        holdout = pd.DataFrame({"genreSimilarity": [-1.0, 0.1, 1.0]})

        result = assign_similarity_bands(holdout, rule)

        self.assertEqual(result.tolist(), ["high_conflict", "mixed", "similar"])
        self.assertEqual(rule["cutoffs_source"], "development pairs only")


if __name__ == "__main__":
    unittest.main()
