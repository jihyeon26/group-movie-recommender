"""Tests for biased explicit-rating matrix factorization."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from group_movie_recommender.algorithms.explicit_mf import (
    ExplicitMFConfig,
    build_explicit_rating_data,
    score_explicit_mf_users,
    train_explicit_mf,
)


class ExplicitMFTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ratings = pd.DataFrame({
            "userId": [1, 1, 2, 2, 3, 3],
            "movieId": [10, 20, 10, 30, 20, 30],
            "rating": [5.0, 1.0, 4.5, 2.0, 1.5, 5.0],
        })
        self.data = build_explicit_rating_data(
            self.ratings, np.array([1, 2, 3]), np.array([10, 20, 30])
        )

    def test_training_is_reproducible_and_scores_full_catalogue(self) -> None:
        config = ExplicitMFConfig(
            factors=4, epochs=3, batch_size=3, patience=3, random_seed=7,
        )
        first = train_explicit_mf(self.data, config)
        second = train_explicit_mf(self.data, config)
        scores = score_explicit_mf_users(first.artifact, np.array([1, 3]))

        np.testing.assert_allclose(
            first.artifact["user_factors"], second.artifact["user_factors"]
        )
        self.assertEqual(scores.shape, (2, 3))
        self.assertTrue(np.isfinite(scores).all())
        self.assertLess(
            first.history.iloc[-1]["training_rmse"],
            first.history.iloc[0]["training_rmse"],
        )

    def test_validation_restores_best_epoch(self) -> None:
        observed = []

        def validate(_artifact, epoch):
            observed.append(epoch)
            return (1.0 if epoch == 1 else 0.0, 0.0)

        result = train_explicit_mf(
            self.data,
            ExplicitMFConfig(
                factors=2, epochs=5, batch_size=3, patience=2, random_seed=11,
            ),
            validation_callback=validate,
        )

        self.assertEqual(result.best_epoch, 1)
        self.assertEqual(result.stopped_epoch, 3)
        self.assertTrue(result.early_stopped)
        self.assertEqual(observed, [1, 2, 3])

    def test_low_ratings_are_retained(self) -> None:
        self.assertIn(1.0, self.data.ratings.tolist())
        self.assertIn(1.5, self.data.ratings.tolist())


if __name__ == "__main__":
    unittest.main()
