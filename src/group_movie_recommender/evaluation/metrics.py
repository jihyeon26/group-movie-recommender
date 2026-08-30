"""Relevance sets and ranking metrics for shared two-user recommendation lists."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


def recall_at_k(
    ranked_movie_ids: Iterable[int],
    relevant_movie_ids: set[int],
    *,
    k: int = 10,
) -> float:
    """Return the fraction of relevant movies retrieved in the first k positions."""

    if k <= 0:
        raise ValueError("k must be a positive integer")
    if not relevant_movie_ids:
        return float("nan")

    top_k = list(ranked_movie_ids)[:k]
    hits = sum(int(movie_id) in relevant_movie_ids for movie_id in top_k)
    return float(hits / len(relevant_movie_ids))


def ndcg_at_k(
    ranked_movie_ids: Iterable[int],
    relevant_movie_ids: set[int],
    *,
    k: int = 10,
) -> float:
    """Compute binary normalized discounted cumulative gain at rank k."""

    if k <= 0:
        raise ValueError("k must be a positive integer")
    if not relevant_movie_ids:
        return float("nan")

    top_k = list(ranked_movie_ids)[:k]
    gains = np.fromiter(
        (int(movie_id) in relevant_movie_ids for movie_id in top_k),
        dtype=np.float64,
    )
    discounts = np.log2(np.arange(2, len(top_k) + 2, dtype=np.float64))
    dcg = float(np.sum(gains / discounts))

    ideal_length = min(k, len(relevant_movie_ids))
    ideal_discounts = np.log2(np.arange(2, ideal_length + 2, dtype=np.float64))
    idcg = float(np.sum(1.0 / ideal_discounts))
    return dcg / idcg


def build_relevant_item_sets(
    positive_events: pd.DataFrame,
    eligible_movie_ids: set[int],
) -> dict[int, set[int]]:
    """Build per-user ground truth restricted to recommendable catalogue items."""

    eligible_events = positive_events.loc[
        positive_events["movieId"].isin(eligible_movie_ids),
        ["userId", "movieId"],
    ].drop_duplicates()
    return {
        int(user_id): set(group["movieId"].astype(int))
        for user_id, group in eligible_events.groupby("userId", sort=False)
    }


def build_pair_relevant_item_sets(
    pairs: pd.DataFrame,
    positive_events: pd.DataFrame,
    eligible_movie_ids: set[int],
    seen_items: dict[int, set[int]],
) -> dict[tuple[int, int], set[int]]:
    """Build relevance after applying each pair's union-of-seen candidate rule."""

    positive_sets = build_relevant_item_sets(positive_events, eligible_movie_ids)
    result: dict[tuple[int, int], set[int]] = {}
    for fallback_pair_id, row in enumerate(pairs.itertuples(index=False)):
        pair_id = int(getattr(row, "pairId", fallback_pair_id))
        user_a = int(row.userA)
        user_b = int(row.userB)
        available = eligible_movie_ids - (
            seen_items.get(user_a, set()) | seen_items.get(user_b, set())
        )
        result[(pair_id, user_a)] = positive_sets.get(user_a, set()) & available
        result[(pair_id, user_b)] = positive_sets.get(user_b, set()) & available
    return result


def evaluate_shared_rankings(
    recommendations: pd.DataFrame,
    relevant_items: Mapping[int | tuple[int, int], set[int]],
    *,
    catalog_size: int,
    k: int = 10,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate the same ranked list against each member's held-out positives."""

    if catalog_size <= 0:
        raise ValueError("catalog_size must be a positive integer")

    required = {"pairId", "userA", "userB", "rank", "movieId"}
    missing = required - set(recommendations.columns)
    if missing:
        raise ValueError(f"Recommendations are missing columns: {sorted(missing)}")

    rows: list[dict[str, int | float]] = []
    recommended_items: set[int] = set()
    skipped_pairs = 0

    ordered = recommendations.sort_values(["pairId", "rank"])
    for pair_id, group in ordered.groupby("pairId", sort=False):
        first = group.iloc[0]
        user_a = int(first["userA"])
        user_b = int(first["userB"])
        positives_a = relevant_items.get(
            (int(pair_id), user_a), relevant_items.get(user_a, set())
        )
        positives_b = relevant_items.get(
            (int(pair_id), user_b), relevant_items.get(user_b, set())
        )

        if not positives_a or not positives_b:
            skipped_pairs += 1
            continue

        ranked_items = group.loc[group["rank"] <= k, "movieId"].astype(int).tolist()
        recommended_items.update(ranked_items)

        ndcg_a = ndcg_at_k(ranked_items, positives_a, k=k)
        ndcg_b = ndcg_at_k(ranked_items, positives_b, k=k)
        recall_a = recall_at_k(ranked_items, positives_a, k=k)
        recall_b = recall_at_k(ranked_items, positives_b, k=k)

        rows.append(
            {
                "pairId": int(pair_id),
                "userA": user_a,
                "userB": user_b,
                f"ndcgA@{k}": ndcg_a,
                f"ndcgB@{k}": ndcg_b,
                f"averageNDCG@{k}": (ndcg_a + ndcg_b) / 2.0,
                f"minimumNDCG@{k}": min(ndcg_a, ndcg_b),
                f"ndcgGap@{k}": abs(ndcg_a - ndcg_b),
                f"recallA@{k}": recall_a,
                f"recallB@{k}": recall_b,
                f"averageRecall@{k}": (recall_a + recall_b) / 2.0,
                f"minimumRecall@{k}": min(recall_a, recall_b),
            }
        )

    pair_metrics = pd.DataFrame.from_records(rows)
    if pair_metrics.empty:
        raise ValueError("No pair has held-out relevant movies for both members")

    summary: dict[str, Any] = {
        "k": k,
        "evaluated_pairs": int(len(pair_metrics)),
        "skipped_pairs_without_two_sided_relevance": int(skipped_pairs),
        f"meanAverageNDCG@{k}": float(pair_metrics[f"averageNDCG@{k}"].mean()),
        f"meanMinimumNDCG@{k}": float(pair_metrics[f"minimumNDCG@{k}"].mean()),
        f"meanNDCGGap@{k}": float(pair_metrics[f"ndcgGap@{k}"].mean()),
        f"meanAverageRecall@{k}": float(pair_metrics[f"averageRecall@{k}"].mean()),
        f"meanMinimumRecall@{k}": float(pair_metrics[f"minimumRecall@{k}"].mean()),
        f"catalogueCoverage@{k}": float(len(recommended_items) / catalog_size),
        f"uniqueRecommendedItems@{k}": int(len(recommended_items)),
    }
    return pair_metrics, summary
