"""Tabular summaries for staged LightGCN validation experiments."""

from __future__ import annotations

from typing import Any


def build_stage_rows(
    stage: str,
    training_summary: dict[str, Any],
    validation_report: dict[str, Any],
) -> list[dict[str, Any]]:
    """Flatten one stage into one comparison row per ranking method."""

    subset = training_summary["subset"]
    training = training_summary["training"]
    shared = {
        "stage": stage,
        "pair_limit": int(subset["pair_limit"]),
        "total_user_limit": int(subset["total_user_limit"]),
        "selected_pair_count": int(validation_report["pairs"]),
        "graph_user_count": int(subset["graph_users"]),
        "graph_movie_count": int(subset["graph_movies"]),
        "positive_edge_count": int(subset["positive_edges"]),
        "training_steps": int(training["steps"]),
        "final_training_bpr_loss": float(
            training_summary["final_fixed_training_bpr_loss"]
        ),
        "validation_target_coverage": float(
            validation_report["pair_validation_target_coverage"]
        ),
    }
    rows: list[dict[str, Any]] = []
    for method in validation_report["methods"]:
        rows.append(
            {
                **shared,
                "method": method["method"],
                "conflict_weight": method.get("conflict_weight"),
                "evaluated_pair_count": int(method["evaluated_pairs"]),
                "skipped_pair_count": int(
                    method["skipped_pairs_without_two_sided_relevance"]
                ),
                "mean_minimum_ndcg_at_10": float(method["meanMinimumNDCG@10"]),
                "mean_average_ndcg_at_10": float(method["meanAverageNDCG@10"]),
                "mean_ndcg_gap_at_10": float(method["meanNDCGGap@10"]),
                "mean_average_recall_at_10": float(method["meanAverageRecall@10"]),
                "catalogue_coverage_at_10": float(method["catalogueCoverage@10"]),
                "two_sided_hit_pair_rate": float(method["twoSidedHitPairRate"]),
                "one_sided_hit_pair_rate": float(method["oneSidedHitPairRate"]),
                "no_hit_pair_rate": float(method["noHitPairRate"]),
            }
        )
    return rows
