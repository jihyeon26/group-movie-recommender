"""Tests for the popularity baseline."""

from __future__ import annotations

import unittest

import pandas as pd

from group_movie_recommender.popularity import (
    PopularityRecommender,
    recommend_pairs_by_popularity,
)


class PopularityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.warm_catalog = pd.DataFrame(
            {
                "movieId": [30, 10, 20, 40],
                "trainPositiveCount": [5, 100, 50, 5],
            }
        )

    def test_ranking_and_seen_item_filter(self) -> None:
        model = PopularityRecommender.from_warm_catalog(self.warm_catalog)

        self.assertEqual(model.ranked_movie_ids.tolist(), [10, 20, 30, 40])
        recommendations = model.recommend({10}, {20}, k=2)
        self.assertEqual(recommendations, [(30, 5.0), (40, 5.0)])

    def test_pair_recommendation_schema(self) -> None:
        pairs = pd.DataFrame({"userA": [1], "userB": [2]})
        seen_items = {1: {10}, 2: {30}}

        result = recommend_pairs_by_popularity(
            pairs,
            self.warm_catalog,
            seen_items,
            k=2,
        )

        self.assertEqual(result["movieId"].tolist(), [20, 40])
        self.assertEqual(result["rank"].tolist(), [1, 2])
        self.assertEqual(result[["userA", "userB"]].iloc[0].tolist(), [1, 2])


if __name__ == "__main__":
    unittest.main()
