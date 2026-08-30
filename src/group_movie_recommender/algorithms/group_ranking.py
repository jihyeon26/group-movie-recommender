"""Average and conflict-aware aggregation of two users' preference scores."""

from __future__ import annotations

import numpy as np
import pandas as pd


NORMALIZED_SCORE_COLUMNS = ("qA", "qB")


def normalize_member_scores(
    candidate_scores: pd.DataFrame,
    *,
    score_a_column: str = "scoreA",
    score_b_column: str = "scoreB",
) -> pd.DataFrame:
    """Convert each member's raw candidate scores to percentile scores in (0, 1]."""

    required = {"movieId", score_a_column, score_b_column}
    missing = required - set(candidate_scores.columns)
    if missing:
        raise ValueError(f"Candidate scores are missing columns: {sorted(missing)}")
    if candidate_scores["movieId"].duplicated().any():
        raise ValueError("movieId must be unique within one pair's candidate set")

    result = candidate_scores.copy()
    result["qA"] = result[score_a_column].rank(method="average", pct=True)
    result["qB"] = result[score_b_column].rank(method="average", pct=True)
    return result


def score_group_candidates(
    normalized_scores: pd.DataFrame,
    *,
    conflict_weight: float,
) -> pd.DataFrame:
    """Blend average satisfaction with the less-satisfied member's score.

    A conflict weight of zero is the average-score baseline. A weight of one is
    least misery, where only the lower member score determines the group score.
    """

    if not 0.0 <= conflict_weight <= 1.0:
        raise ValueError("conflict_weight must be between 0 and 1")

    required = {"movieId", *NORMALIZED_SCORE_COLUMNS}
    missing = required - set(normalized_scores.columns)
    if missing:
        raise ValueError(f"Normalized scores are missing columns: {sorted(missing)}")
    if normalized_scores["movieId"].duplicated().any():
        raise ValueError("movieId must be unique within one pair's candidate set")

    values = normalized_scores.loc[:, NORMALIZED_SCORE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Normalized scores must be finite")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("qA and qB must be between 0 and 1")

    result = normalized_scores.copy()
    result["averageScore"] = result[["qA", "qB"]].mean(axis=1)
    result["minimumScore"] = result[["qA", "qB"]].min(axis=1)
    result["disagreement"] = (result["qA"] - result["qB"]).abs()
    result["groupScore"] = (
        (1.0 - conflict_weight) * result["averageScore"]
        + conflict_weight * result["minimumScore"]
    )
    return result


def rank_group_candidates(
    normalized_scores: pd.DataFrame,
    *,
    conflict_weight: float,
    k: int = 10,
) -> pd.DataFrame:
    """Return a deterministic shared Top-k list for one two-user pair."""

    if k <= 0:
        raise ValueError("k must be a positive integer")

    scored = score_group_candidates(
        normalized_scores,
        conflict_weight=conflict_weight,
    )
    ranked = (
        scored.sort_values(
            ["groupScore", "minimumScore", "averageScore", "movieId"],
            ascending=[False, False, False, True],
        )
        .head(k)
        .reset_index(drop=True)
    )
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1, dtype=np.int32))
    return ranked
