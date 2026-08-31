"""Tests for implicit item-neighborhood fitting and scoring."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from group_movie_recommender.algorithms.graph_data import build_bipartite_graph
from group_movie_recommender.algorithms.item_knn import (
    fit_item_knn,
    score_item_knn_profile,
    score_item_knn_users,
)


class ItemKNNTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = build_bipartite_graph(pd.DataFrame({
            "userId": [1, 1, 2, 2, 3, 3],
            "movieId": [10, 20, 10, 30, 20, 40],
        }))

    def test_cosine_neighbors_are_deterministic_and_exclude_self(self) -> None:
        first = fit_item_knn(self.graph, max_neighbors=3)
        second = fit_item_knn(self.graph, max_neighbors=3)
        np.testing.assert_array_equal(first.neighbor_indices, second.neighbor_indices)
        np.testing.assert_allclose(first.neighbor_scores, second.neighbor_scores)
        for source, neighbors in enumerate(first.neighbor_indices):
            self.assertNotIn(source, neighbors.tolist())

    def test_user_receives_scores_from_each_seen_movie(self) -> None:
        model = fit_item_knn(self.graph, max_neighbors=3)
        scores = score_item_knn_users(model, self.graph, np.array([1]), neighbors=3)
        movie_30 = self.graph.movie_indices(np.array([30]))[0]
        movie_40 = self.graph.movie_indices(np.array([40]))[0]
        self.assertGreater(scores[0, movie_30], 0.0)
        self.assertGreater(scores[0, movie_40], 0.0)

    def test_shrinkage_reduces_similarity(self) -> None:
        plain = fit_item_knn(self.graph, max_neighbors=3, shrinkage=0.0)
        shrunk = fit_item_knn(self.graph, max_neighbors=3, shrinkage=10.0)
        self.assertLess(shrunk.neighbor_scores.max(), plain.neighbor_scores.max())

    def test_external_profile_scores_neighbors_of_positive_movies(self) -> None:
        model = fit_item_knn(self.graph, max_neighbors=3)
        scores = score_item_knn_profile(model, np.array([10]), neighbors=3)
        movie_30 = self.graph.movie_indices(np.array([30]))[0]
        movie_40 = self.graph.movie_indices(np.array([40]))[0]
        self.assertGreater(scores[movie_30], 0.0)
        self.assertEqual(scores[movie_40], 0.0)


if __name__ == "__main__":
    unittest.main()
