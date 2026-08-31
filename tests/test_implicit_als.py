"""Tests for confidence-weighted implicit ALS and new-user folding."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from group_movie_recommender.algorithms.graph_data import build_bipartite_graph
from group_movie_recommender.algorithms.implicit_als import (
    ImplicitALSConfig, recalculate_user_factor, train_implicit_als,
)


class ImplicitALSTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = build_bipartite_graph(pd.DataFrame({
            "userId": [1, 1, 2, 2, 3, 3],
            "movieId": [10, 20, 20, 30, 10, 30],
        }))

    def test_training_is_reproducible_and_shapes_match_graph(self) -> None:
        config = ImplicitALSConfig(factors=3, iterations=3, batch_size=2, random_seed=7)
        first = train_implicit_als(self.graph, config)
        second = train_implicit_als(self.graph, config)
        np.testing.assert_allclose(first.user_factors, second.user_factors)
        np.testing.assert_allclose(first.movie_factors, second.movie_factors)
        self.assertEqual(first.user_factors.shape, (3, 3))
        self.assertEqual(first.movie_factors.shape, (3, 3))

    def test_validation_restores_best_iteration(self) -> None:
        snapshots = {}

        def validate(users, movies, iteration):
            snapshots[iteration] = (users.copy(), movies.copy())
            return ({1: 0.4, 2: 0.5, 3: 0.3, 4: 0.2}[iteration], 0.0)

        result = train_implicit_als(
            self.graph,
            ImplicitALSConfig(factors=2, iterations=5, batch_size=2, patience=2),
            validation_callback=validate,
        )
        self.assertEqual(result.best_iteration, 2)
        self.assertEqual(result.stopped_iteration, 4)
        self.assertTrue(result.early_stopped)
        np.testing.assert_allclose(result.user_factors, snapshots[2][0])
        np.testing.assert_allclose(result.movie_factors, snapshots[2][1])

    def test_new_user_factor_has_expected_dimension_and_finite_values(self) -> None:
        config = ImplicitALSConfig(factors=3, iterations=2, batch_size=2)
        result = train_implicit_als(self.graph, config)
        user = recalculate_user_factor(result.movie_factors, np.array([0, 1]), config)
        self.assertEqual(user.shape, (3,))
        self.assertTrue(np.isfinite(user).all())


if __name__ == "__main__":
    unittest.main()
