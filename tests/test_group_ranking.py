"""Tests for average and conflict-aware two-user ranking."""

from __future__ import annotations

import unittest

import pandas as pd

from group_movie_recommender.algorithms.group_ranking import (
    normalize_member_scores,
    rank_group_candidates,
    score_group_candidates,
)


class GroupRankingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.normalized_scores = pd.DataFrame(
            {
                "movieId": [101, 102, 103],
                "qA": [1.0, 0.6, 0.3],
                "qB": [0.4, 0.6, 0.9],
            }
        )

    def test_conflict_weight_endpoints(self) -> None:
        average = score_group_candidates(
            self.normalized_scores,
            conflict_weight=0.0,
        )
        least_misery = score_group_candidates(
            self.normalized_scores,
            conflict_weight=1.0,
        )

        self.assertEqual(average["groupScore"].tolist(), [0.7, 0.6, 0.6])
        self.assertEqual(least_misery["groupScore"].tolist(), [0.4, 0.6, 0.3])

    def test_conflict_awareness_promotes_a_compromise(self) -> None:
        average_ranking = rank_group_candidates(
            self.normalized_scores,
            conflict_weight=0.0,
            k=3,
        )
        conflict_ranking = rank_group_candidates(
            self.normalized_scores,
            conflict_weight=0.75,
            k=3,
        )

        self.assertEqual(average_ranking["movieId"].tolist(), [101, 102, 103])
        self.assertEqual(conflict_ranking["movieId"].tolist(), [102, 101, 103])

    def test_raw_scores_are_normalized_per_member(self) -> None:
        raw_scores = pd.DataFrame(
            {
                "movieId": [10, 20, 30],
                "scoreA": [100.0, 20.0, 5.0],
                "scoreB": [-2.0, 4.0, 1.0],
            }
        )
        normalized = normalize_member_scores(raw_scores)

        self.assertEqual(normalized["qA"].tolist(), [1.0, 2 / 3, 1 / 3])
        self.assertEqual(normalized["qB"].tolist(), [1 / 3, 1.0, 2 / 3])

    def test_invalid_conflict_weight_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            rank_group_candidates(
                self.normalized_scores,
                conflict_weight=1.1,
            )


if __name__ == "__main__":
    unittest.main()
