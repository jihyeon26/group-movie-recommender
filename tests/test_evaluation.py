"""Tests for individual and two-user group ranking metrics."""

from __future__ import annotations

import math
import unittest

import pandas as pd

from group_movie_recommender.evaluation import (
    build_relevant_item_sets,
    evaluate_shared_rankings,
    ndcg_at_k,
    recall_at_k,
)


class EvaluationTests(unittest.TestCase):
    def test_individual_metrics(self) -> None:
        ranking = [10, 20, 30]
        relevant = {10, 30}

        self.assertEqual(recall_at_k(ranking, relevant, k=3), 1.0)
        expected_ndcg = (1.0 + 1.0 / math.log2(4)) / (
            1.0 + 1.0 / math.log2(3)
        )
        self.assertAlmostEqual(ndcg_at_k(ranking, relevant, k=3), expected_ndcg)

    def test_shared_ranking_aggregates_member_metrics(self) -> None:
        recommendations = pd.DataFrame(
            {
                "pairId": [0, 0, 0],
                "userA": [1, 1, 1],
                "userB": [2, 2, 2],
                "rank": [1, 2, 3],
                "movieId": [10, 20, 30],
            }
        )
        relevant = {1: {10, 30}, 2: {20}}

        pair_metrics, summary = evaluate_shared_rankings(
            recommendations,
            relevant,
            catalog_size=5,
            k=3,
        )

        ndcg_a = ndcg_at_k([10, 20, 30], {10, 30}, k=3)
        ndcg_b = ndcg_at_k([10, 20, 30], {20}, k=3)
        self.assertAlmostEqual(pair_metrics.loc[0, "averageNDCG@3"], (ndcg_a + ndcg_b) / 2)
        self.assertAlmostEqual(pair_metrics.loc[0, "minimumNDCG@3"], min(ndcg_a, ndcg_b))
        self.assertAlmostEqual(summary["catalogueCoverage@3"], 3 / 5)

    def test_relevance_is_restricted_to_eligible_movies(self) -> None:
        positives = pd.DataFrame(
            {"userId": [1, 1, 2], "movieId": [10, 99, 20]}
        )
        result = build_relevant_item_sets(positives, {10, 20})
        self.assertEqual(result, {1: {10}, 2: {20}})


if __name__ == "__main__":
    unittest.main()
