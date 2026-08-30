"""Tests for staged experiment summaries."""

from __future__ import annotations

import unittest

from group_movie_recommender.evaluation.experiment_summary import build_stage_rows


class ExperimentSummaryTests(unittest.TestCase):
    def test_one_row_is_created_per_method(self) -> None:
        training = {
            "subset": {
                "pair_limit": 25,
                "total_user_limit": 500,
                "graph_users": 500,
                "graph_movies": 6000,
                "positive_edges": 40000,
            },
            "training": {"steps": 30},
            "final_fixed_training_bpr_loss": 0.3,
        }
        validation = {
            "pairs": 25,
            "pair_validation_target_coverage": 0.6,
            "methods": [
                {
                    "method": "popularity",
                    "evaluated_pairs": 24,
                    "skipped_pairs_without_two_sided_relevance": 1,
                    "meanMinimumNDCG@10": 0.01,
                    "meanAverageNDCG@10": 0.02,
                    "meanNDCGGap@10": 0.02,
                    "meanAverageRecall@10": 0.03,
                    "catalogueCoverage@10": 0.04,
                },
                {
                    "method": "lightgcn_conflict_0.5",
                    "conflict_weight": 0.5,
                    "evaluated_pairs": 24,
                    "skipped_pairs_without_two_sided_relevance": 1,
                    "meanMinimumNDCG@10": 0.02,
                    "meanAverageNDCG@10": 0.03,
                    "meanNDCGGap@10": 0.01,
                    "meanAverageRecall@10": 0.04,
                    "catalogueCoverage@10": 0.05,
                },
            ],
        }

        rows = build_stage_rows("smoke", training, validation)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["stage"], "smoke")
        self.assertEqual(rows[1]["conflict_weight"], 0.5)
        self.assertEqual(rows[1]["validation_target_coverage"], 0.6)
        self.assertEqual(rows[1]["evaluated_pair_count"], 24)


if __name__ == "__main__":
    unittest.main()
