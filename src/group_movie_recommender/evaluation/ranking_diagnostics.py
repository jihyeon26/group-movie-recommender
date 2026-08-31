"""Diagnostics for how aggregation methods change shared ranked lists."""

from __future__ import annotations

import pandas as pd


def compare_rankings_to_reference(
    recommendations: pd.DataFrame,
    *,
    reference_method: str,
    k: int,
) -> pd.DataFrame:
    """Measure Top-k overlap and position changes relative to one method."""

    required = {"method", "pairId", "rank", "movieId"}
    missing = required - set(recommendations.columns)
    if missing:
        raise ValueError(f"Recommendations are missing columns: {sorted(missing)}")
    if k <= 0:
        raise ValueError("k must be a positive integer")

    top_k = recommendations.loc[recommendations["rank"] <= k].sort_values(
        ["method", "pairId", "rank"]
    )
    rankings = {
        str(method): {
            int(pair_id): tuple(group["movieId"].astype(int))
            for pair_id, group in method_rows.groupby("pairId", sort=False)
        }
        for method, method_rows in top_k.groupby("method", sort=False)
    }
    if reference_method not in rankings:
        raise ValueError(f"Reference method is absent: {reference_method}")

    reference = rankings[reference_method]
    records: list[dict[str, object]] = []
    for method, candidate in rankings.items():
        common_pairs = sorted(reference.keys() & candidate.keys())
        identical: list[float] = []
        overlaps: list[float] = []
        position_agreements: list[float] = []
        for pair_id in common_pairs:
            reference_items = reference[pair_id]
            candidate_items = candidate[pair_id]
            identical.append(float(reference_items == candidate_items))
            denominator = min(len(reference_items), len(candidate_items))
            if denominator == 0:
                continue
            overlaps.append(
                len(set(reference_items) & set(candidate_items)) / denominator
            )
            position_agreements.append(
                sum(
                    left == right
                    for left, right in zip(reference_items, candidate_items)
                )
                / denominator
            )
        records.append(
            {
                "referenceMethod": reference_method,
                "method": method,
                "comparedPairs": len(common_pairs),
                "identicalTopKPairRate": _mean(identical),
                "meanItemOverlapAtK": _mean(overlaps),
                "meanPositionAgreementAtK": _mean(position_agreements),
            }
        )
    return pd.DataFrame(records)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
