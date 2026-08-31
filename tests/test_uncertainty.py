"""Tests for paired bootstrap ranking comparisons."""

from __future__ import annotations

import unittest

import pandas as pd

from group_movie_recommender.evaluation.uncertainty import (
    paired_bootstrap_mean_difference,
)


class UncertaintyTests(unittest.TestCase):
    def test_paired_difference_uses_only_common_pairs(self) -> None:
        pair_metrics = pd.DataFrame(
            {
                "method": ["a", "a", "a", "b", "b"],
                "pairId": [1, 2, 3, 1, 2],
                "minimumNDCG@10": [0.4, 0.6, 1.0, 0.1, 0.2],
            }
        )

        result = paired_bootstrap_mean_difference(
            pair_metrics,
            method_a="a",
            method_b="b",
            metric="minimumNDCG@10",
            n_resamples=200,
            random_seed=7,
        )

        self.assertEqual(result["commonPairs"], 2)
        self.assertAlmostEqual(result["meanDifferenceAminusB"], 0.35)
        self.assertEqual(result["bootstrapProbabilityAboveZero"], 1.0)


if __name__ == "__main__":
    unittest.main()
