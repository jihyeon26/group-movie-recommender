"""Tests for the declared model-selection order."""

from __future__ import annotations

import unittest

from group_movie_recommender.pipelines.validation_experiment import selection_key


class ValidationExperimentTests(unittest.TestCase):
    def test_minimum_member_metric_precedes_average_metric(self) -> None:
        fairer = {"meanMinimumNDCG@10": 0.02, "meanAverageNDCG@10": 0.03}
        higher_average = {"meanMinimumNDCG@10": 0.01, "meanAverageNDCG@10": 0.20}
        self.assertGreater(selection_key(fairer, 10), selection_key(higher_average, 10))


if __name__ == "__main__":
    unittest.main()
