"""Tests for framework-independent LightGCN mathematics."""

from __future__ import annotations

import math
import unittest

import numpy as np

from group_movie_recommender.algorithms.lightgcn_math import (
    bpr_loss,
    lightgcn_propagate,
    normalized_edge_weights,
    propagate_once,
    score_user_movie_pairs,
    split_user_movie_embeddings,
)


class LightGCNMathTests(unittest.TestCase):
    def test_symmetric_degree_normalization(self) -> None:
        edge_index = np.array([[0, 0, 1, 2], [1, 2, 0, 0]], dtype=np.int64)
        weights = normalized_edge_weights(edge_index, num_nodes=3)
        expected = np.full(4, 1.0 / math.sqrt(2.0))
        np.testing.assert_allclose(weights, expected)

    def test_one_layer_aggregates_weighted_neighbors(self) -> None:
        edge_index = np.array([[0, 0, 1, 2], [1, 2, 0, 0]], dtype=np.int64)
        embeddings = np.array([[1.0], [2.0], [4.0]])
        propagated = propagate_once(embeddings, edge_index)
        expected = np.array(
            [
                [(2.0 + 4.0) / math.sqrt(2.0)],
                [1.0 / math.sqrt(2.0)],
                [1.0 / math.sqrt(2.0)],
            ]
        )
        np.testing.assert_allclose(propagated, expected)

    def test_lightgcn_averages_initial_and_propagated_embeddings(self) -> None:
        edge_index = np.array([[0, 1], [1, 0]], dtype=np.int64)
        initial = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = lightgcn_propagate(initial, edge_index, num_layers=1)

        np.testing.assert_allclose(
            result.final_embeddings,
            np.array([[0.5, 0.5], [0.5, 0.5]]),
        )
        self.assertEqual(len(result.layer_embeddings), 2)

    def test_pair_scoring_and_bpr_loss(self) -> None:
        node_embeddings = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [2.0, 0.0],
                [0.0, 0.5],
            ]
        )
        users, movies = split_user_movie_embeddings(node_embeddings, num_users=2)
        positive_scores = score_user_movie_pairs(
            users,
            movies,
            np.array([0, 1]),
            np.array([0, 1]),
        )
        negative_scores = score_user_movie_pairs(
            users,
            movies,
            np.array([0, 1]),
            np.array([1, 0]),
        )

        np.testing.assert_allclose(positive_scores, [2.0, 0.5])
        np.testing.assert_allclose(negative_scores, [0.0, 0.0])
        self.assertLess(bpr_loss(positive_scores, negative_scores), math.log(2.0))
        self.assertGreater(bpr_loss(negative_scores, positive_scores), math.log(2.0))


if __name__ == "__main__":
    unittest.main()
