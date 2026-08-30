"""Tests for filtering and pair ranking from exported embeddings."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from group_movie_recommender.algorithms.embedding_inference import recommend_pairs_from_embeddings


class EmbeddingInferenceTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
