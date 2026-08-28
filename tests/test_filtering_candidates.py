"""Tests for warm-item and pair candidate filtering."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from group_movie_recommender.candidates import (
    add_joint_test_counts,
    build_seen_item_sets,
    candidate_movies_for_pair,
    keep_first_k_interactions,
)
from group_movie_recommender.config import PreprocessingConfig
from group_movie_recommender.filtering import build_warm_catalog, filter_edges_to_catalog


class FilteringAndCandidateTests(unittest.TestCase):
    def test_warm_catalog_and_candidates(self) -> None:
        config = PreprocessingConfig(warm_item_min_train_positives=2)
        edges = pd.DataFrame(
            {
                "userId": [1, 2, 1, 2, 3],
                "movieId": [10, 10, 20, 20, 30],
            }
        )
        warm = build_warm_catalog(edges, config)
        self.assertEqual(set(warm["movieId"]), {10, 20})
        self.assertEqual(len(filter_edges_to_catalog(edges, warm)), 4)

        ratings = pd.DataFrame(
            {
                "userId": [1, 1, 2, 2],
                "movieId": [10, 30, 20, 40],
                "rating": [4.0, 3.0, 5.0, 2.0],
                "timestamp": [1, 2, 1, 2],
                "split": ["train", "validation", "train", "test"],
            }
        )
        seen = build_seen_item_sets(ratings, np.array([1, 2], dtype=np.int32))
        candidates = candidate_movies_for_pair(np.array([10, 20, 30, 50]), seen, 1, 2)
        self.assertEqual(candidates.tolist(), [50])

    def test_joint_counts_and_sparse_history(self) -> None:
        pairs = pd.DataFrame({"userA": [1], "userB": [2]})
        positives = pd.DataFrame(
            {"userId": [1, 1, 2, 2], "movieId": [10, 20, 10, 30]}
        )
        result = add_joint_test_counts(pairs, positives)
        self.assertEqual(int(result.loc[0, "jointTestPositiveCount"]), 1)

        history = pd.DataFrame(
            {
                "userId": [1, 1, 1, 2, 2],
                "movieId": [30, 20, 10, 50, 40],
                "timestamp": [3, 2, 1, 2, 1],
            }
        )
        truncated = keep_first_k_interactions(history, 2)
        self.assertEqual(truncated.loc[truncated["userId"] == 1, "movieId"].tolist(), [10, 20])


if __name__ == "__main__":
    unittest.main()

