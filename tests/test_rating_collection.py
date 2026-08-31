"""Tests for one-person MovieLens-format rating exports."""

from __future__ import annotations

import unittest

from group_movie_recommender.preprocessing.rating_collection import (
    build_movielens_ratings,
)


class RatingCollectionTests(unittest.TestCase):
    def test_exact_schema_values_and_sort_order(self) -> None:
        records = {
            20: {"rating": 4.5, "timestamp": 1_700_000_020},
            10: {"rating": 3.0, "timestamp": 1_700_000_010},
        }
        frame = build_movielens_ratings(records, user_id=1_000_000_001)

        self.assertEqual(
            frame.columns.tolist(),
            ["userId", "movieId", "rating", "timestamp"],
        )
        self.assertEqual(frame["movieId"].tolist(), [10, 20])
        self.assertEqual(frame["userId"].tolist(), [1_000_000_001, 1_000_000_001])
        self.assertEqual(frame["rating"].tolist(), [3.0, 4.5])

    def test_empty_export_keeps_exact_schema(self) -> None:
        frame = build_movielens_ratings({}, user_id=1_000_000_001)
        self.assertEqual(
            frame.columns.tolist(),
            ["userId", "movieId", "rating", "timestamp"],
        )
        self.assertTrue(frame.empty)

    def test_invalid_half_star_rating_is_rejected(self) -> None:
        records = {10: {"rating": 4.2, "timestamp": 1_700_000_010}}
        with self.assertRaises(ValueError):
            build_movielens_ratings(records, user_id=1_000_000_001)


if __name__ == "__main__":
    unittest.main()
