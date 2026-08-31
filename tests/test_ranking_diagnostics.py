"""Tests for cross-method ranking sensitivity diagnostics."""

from __future__ import annotations

import unittest

import pandas as pd

from group_movie_recommender.evaluation.ranking_diagnostics import (
    compare_rankings_to_reference,
)


class RankingDiagnosticsTests(unittest.TestCase):
    def test_overlap_and_position_changes_are_measured(self) -> None:
        recommendations = pd.DataFrame(
            {
                "method": ["average"] * 3 + ["conflict"] * 3,
                "pairId": [0] * 6,
                "rank": [1, 2, 3, 1, 2, 3],
                "movieId": [10, 20, 30, 20, 10, 40],
            }
        )

        result = compare_rankings_to_reference(
            recommendations,
            reference_method="average",
            k=3,
        ).set_index("method")

        self.assertEqual(result.loc["average", "identicalTopKPairRate"], 1.0)
        self.assertEqual(result.loc["conflict", "identicalTopKPairRate"], 0.0)
        self.assertAlmostEqual(result.loc["conflict", "meanItemOverlapAtK"], 2 / 3)
        self.assertEqual(result.loc["conflict", "meanPositionAgreementAtK"], 0.0)


if __name__ == "__main__":
    unittest.main()
