"""Tests for explicit-MF conflict-band analysis."""

from __future__ import annotations

import unittest

import pandas as pd

from group_movie_recommender.pipelines.explicit_mf_experiment import (
    add_conflict_bands,
    summarize_conflict_bands,
)


class ExplicitMFExperimentTests(unittest.TestCase):
    def test_conflict_bands_use_training_similarity_features(self) -> None:
        pairs = pd.DataFrame({
            "pairId": [0, 1, 2],
            "userA": [1, 3, 5],
            "userB": [2, 4, 6],
        })
        features = pd.DataFrame({
            "userA": [1, 3, 5],
            "userB": [2, 4, 6],
            "coRatedTrain": [20, 20, 20],
            "genreSimilarity": [0.2, 0.0, -0.2],
            "ratingCorrelation": [0.2, 0.0, -0.2],
        })
        result = add_conflict_bands(pairs, features)

        self.assertEqual(result["conflictBand"].tolist(), ["low", "medium", "high"])
        self.assertTrue(result["conflictScore"].is_monotonic_increasing)

    def test_subgroup_summary_keeps_methods_separate(self) -> None:
        bands = pd.DataFrame({
            "pairId": [0, 1, 2],
            "conflictScore": [0.1, 0.5, 0.9],
            "conflictBand": ["low", "medium", "high"],
        })
        base = pd.DataFrame({
            "pairId": [0, 1, 2],
            "userA": [1, 3, 5],
            "userB": [2, 4, 6],
            "ndcgA@10": [1.0, 0.0, 0.5],
            "ndcgB@10": [0.5, 0.0, 0.5],
            "averageNDCG@10": [0.75, 0.0, 0.5],
            "minimumNDCG@10": [0.5, 0.0, 0.5],
            "ndcgGap@10": [0.5, 0.0, 0.0],
            "averageRecall@10": [0.75, 0.0, 0.5],
            "minimumRecall@10": [0.5, 0.0, 0.5],
        })
        first = base.assign(method="average")
        second = base.assign(method="nash")
        summary = summarize_conflict_bands([first, second], bands, k=10)

        self.assertEqual(len(summary), 6)
        self.assertEqual(set(summary["method"]), {"average", "nash"})


if __name__ == "__main__":
    unittest.main()
