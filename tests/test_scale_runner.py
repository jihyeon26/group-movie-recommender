"""Tests for cross-stage paired model comparisons."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.run_lightgcn_scale_experiment import collect_paired_comparisons


class ScaleRunnerTests(unittest.TestCase):
    def test_cross_stage_comparison_uses_common_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = root / "trained/validation"
            right = root / "bpr_mf/validation"
            left.mkdir(parents=True)
            right.mkdir(parents=True)
            pd.DataFrame(
                {
                    "method": ["lightgcn_average", "lightgcn_average"],
                    "pairId": [1, 2],
                    "minimumNDCG@10": [0.4, 0.2],
                }
            ).to_csv(left / "pair_metrics.csv.gz", index=False)
            pd.DataFrame(
                {
                    "method": ["bpr_mf_average", "bpr_mf_average"],
                    "pairId": [1, 3],
                    "minimumNDCG@10": [0.1, 0.9],
                }
            ).to_csv(right / "pair_metrics.csv.gz", index=False)

            results = collect_paired_comparisons(
                root,
                [
                    {
                        "name": "graph_propagation_average",
                        "method_a": {
                            "stage": "trained",
                            "method": "lightgcn_average",
                        },
                        "method_b": {
                            "stage": "bpr_mf",
                            "method": "bpr_mf_average",
                        },
                        "metric": "minimumNDCG@10",
                    }
                ],
                n_resamples=20,
                random_seed=7,
            )

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["commonPairs"], 1)
            self.assertAlmostEqual(results[0]["meanDifferenceAminusB"], 0.3)


if __name__ == "__main__":
    unittest.main()
