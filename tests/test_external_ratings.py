"""Tests for external rating export normalization."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from group_movie_recommender.preprocessing.external_ratings import (
    load_external_movie_ratings,
)


class ExternalRatingsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.links = pd.DataFrame(
            {"movieId": [10, 20], "imdbId": pd.Series([111, 222], dtype="Int64")}
        )

    def test_imdb_movies_are_mapped_scaled_and_warm_filtered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ratings.csv"
            pd.DataFrame(
                {
                    "Const": ["tt0000111", "tt0000222", "tt0000333"],
                    "Your Rating": [9, 8, 7],
                    "Title Type": ["Movie", "TV Series", "Movie"],
                }
            ).to_csv(path, index=False)

            ratings, summary = load_external_movie_ratings(
                path, self.links, {10}
            )

            self.assertEqual(ratings.to_dict(orient="records"), [{"movieId": 10, "rating": 4.5}])
            self.assertEqual(summary["nonMovieRows"], 1)
            self.assertEqual(summary["unmappedRows"], 1)
            self.assertEqual(summary["usableWarmRatings"], 1)

    def test_movielens_export_is_validated_and_warm_filtered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ratings.csv"
            pd.DataFrame(
                {"movieId": [10, 99], "rating": [4.0, 3.5]}
            ).to_csv(path, index=False)

            ratings, summary = load_external_movie_ratings(
                path, self.links, {10}
            )

            self.assertEqual(ratings.to_dict(orient="records"), [{"movieId": 10, "rating": 4.0}])
            self.assertEqual(summary["coldOrUnavailableMovieRows"], 1)


if __name__ == "__main__":
    unittest.main()
