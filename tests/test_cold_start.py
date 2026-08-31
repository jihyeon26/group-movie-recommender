"""Tests for sparse-profile onboarding and group recommendations."""

from __future__ import annotations

import unittest

import pandas as pd

from group_movie_recommender.algorithms.cold_start import (
    recommend_cold_start_group,
    score_cold_start_candidates,
    select_onboarding_movies,
)


class ColdStartTests(unittest.TestCase):
    def setUp(self) -> None:
        self.movies = pd.DataFrame(
            {
                "movieId": list(range(1, 11)),
                "title": [f"Movie {movie_id}" for movie_id in range(1, 11)],
                "genres": [
                    "Action",
                    "Drama",
                    "Comedy",
                    "Action|Adventure",
                    "Drama|Romance",
                    "Comedy",
                    "Action",
                    "Action",
                    "Action",
                    "Action",
                ],
            }
        )
        self.warm = pd.DataFrame(
            {
                "movieId": list(range(1, 11)),
                "trainPositiveCount": [100, 90, 80, 70, 60, 50, 40, 30, 20, 10],
            }
        )
        self.empty_ratings = pd.DataFrame(columns=["movieId", "rating"])

    def test_onboarding_is_popular_unique_and_genre_diverse(self) -> None:
        selected = select_onboarding_movies(
            self.movies,
            self.warm,
            n=3,
            popular_pool_size=6,
            genres=("Action", "Drama", "Comedy"),
        )

        self.assertEqual(selected["movieId"].tolist(), [1, 2, 3])
        self.assertFalse(selected["movieId"].duplicated().any())

    def test_empty_profile_falls_back_to_popularity(self) -> None:
        scores = score_cold_start_candidates(
            self.movies,
            self.warm,
            self.empty_ratings,
        )

        self.assertEqual(scores.iloc[0]["movieId"], 1)
        self.assertEqual(scores.iloc[0]["profileConfidence"], 0.0)
        self.assertEqual(scores["score"].tolist(), scores["popularityScore"].tolist())

    def test_repeated_action_likes_shift_scores_toward_action(self) -> None:
        ratings = pd.DataFrame(
            {"movieId": [1, 4, 7, 8], "rating": [5.0, 5.0, 5.0, 5.0]}
        )
        scores = score_cold_start_candidates(self.movies, self.warm, ratings)
        fallback = score_cold_start_candidates(
            self.movies, self.warm, self.empty_ratings
        ).set_index("movieId")
        by_id = scores.set_index("movieId")

        self.assertGreater(by_id.loc[9, "tasteScore"], by_id.loc[2, "tasteScore"])
        self.assertGreater(
            by_id.loc[9, "score"] - by_id.loc[2, "score"],
            fallback.loc[9, "score"] - fallback.loc[2, "score"],
        )

    def test_group_list_excludes_every_rating_and_supports_zero_ratings(self) -> None:
        ratings_a = pd.DataFrame({"movieId": [1], "rating": [5.0]})
        ratings_b = pd.DataFrame({"movieId": [2], "rating": [4.0]})

        recommendations = recommend_cold_start_group(
            self.movies,
            self.warm,
            ratings_a,
            ratings_b,
            k=4,
        )
        fallback = recommend_cold_start_group(
            self.movies,
            self.warm,
            self.empty_ratings,
            self.empty_ratings,
            k=3,
        )

        self.assertTrue(set(recommendations["movieId"]).isdisjoint({1, 2}))
        self.assertEqual(recommendations["rank"].tolist(), [1, 2, 3, 4])
        self.assertEqual(fallback["movieId"].tolist(), [1, 2, 3])

    def test_invalid_rating_is_rejected_instead_of_becoming_feedback(self) -> None:
        invalid = pd.DataFrame({"movieId": [1], "rating": [0.0]})
        with self.assertRaises(ValueError):
            score_cold_start_candidates(self.movies, self.warm, invalid)


if __name__ == "__main__":
    unittest.main()
