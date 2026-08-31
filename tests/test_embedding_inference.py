"""Tests for filtering and pair ranking from exported embeddings."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from group_movie_recommender.algorithms.embedding_inference import (
    fold_in_user_embedding,
    recommend_pairs_from_embeddings,
    recommend_pairs_from_score_matrix,
    score_folded_in_pair,
)


class EmbeddingInferenceTests(unittest.TestCase):
    def test_fold_in_is_a_rating_weighted_positive_item_average(self) -> None:
        ratings = pd.DataFrame(
            {"movieId": [10, 20, 30], "rating": [5.0, 4.0, 3.5]}
        )
        movie_ids = np.array([10, 20, 30])
        movie_embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [9.0, 9.0]])

        result = fold_in_user_embedding(ratings, movie_ids, movie_embeddings)

        np.testing.assert_allclose(result, np.array([0.75, 0.25]))

    def test_folded_in_pair_excludes_both_members_seen_items(self) -> None:
        ratings_a = pd.DataFrame({"movieId": [10], "rating": [5.0]})
        ratings_b = pd.DataFrame({"movieId": [20], "rating": [5.0]})
        movie_ids = np.array([10, 20, 30])
        movie_embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])

        result = score_folded_in_pair(
            ratings_a,
            ratings_b,
            movie_ids,
            movie_embeddings,
        )

        self.assertEqual(result["movieId"].tolist(), [30])
        self.assertEqual(result.iloc[0]["scoreA"], 0.5)
        self.assertEqual(result.iloc[0]["scoreB"], 0.5)

    def test_seen_union_is_removed_before_pair_ranking(self) -> None:
        pairs = pd.DataFrame({"pairId": [0], "userA": [10], "userB": [20]})
        user_ids = np.array([10, 20])
        user_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
        movie_ids = np.array([100, 200, 300, 400])
        movie_embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7], [0.2, 0.2]])
        seen = {10: {100}, 20: {200}}

        result = recommend_pairs_from_embeddings(
            pairs,
            user_ids,
            user_embeddings,
            movie_ids,
            movie_embeddings,
            seen,
            conflict_weight=0.75,
            k=3,
        )
        self.assertEqual(result["movieId"].tolist(), [300, 400])
        self.assertTrue(set(result["movieId"]).isdisjoint({100, 200}))

    def test_score_matrix_ranking_removes_seen_union(self) -> None:
        pairs = pd.DataFrame({"pairId": [0], "userA": [10], "userB": [20]})
        result = recommend_pairs_from_score_matrix(
            pairs,
            np.array([10, 20]),
            np.array([[3.0, 1.0, 2.0], [1.0, 3.0, 2.0]]),
            np.array([100, 200, 300]),
            {10: {100}, 20: {200}},
            conflict_weight=0.0,
            k=2,
        )
        self.assertEqual(result["movieId"].tolist(), [300])


if __name__ == "__main__":
    unittest.main()
