"""Period-aware candidate filtering for checkpoint selection and final evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import build_pair_relevant_item_sets
from ..filtering.candidates import build_seen_item_sets
from ..filtering.catalog import positive_events
from ..preprocessing.config import PreprocessingConfig
from ..preprocessing.splitting import add_temporal_split
from ..shared.io import RATING_DTYPES


@dataclass(frozen=True)
class TemporalEvaluationContext:
    pairs: pd.DataFrame
    seen: dict[int, set[int]]
    relevance: dict[tuple[int, int], set[int]]
    catalog_size: int
    diagnostics: dict


def load_focus_ratings(
    ratings_path: Path,
    user_ids: np.ndarray,
    *,
    before_timestamp: int | None = None,
    chunksize: int = 1_000_000,
) -> pd.DataFrame:
    """Keep only focus users and, during selection, strictly pre-test events."""

    frames = []
    for chunk in pd.read_csv(
        ratings_path, usecols=list(RATING_DTYPES), dtype=RATING_DTYPES,
        chunksize=chunksize,
    ):
        mask = chunk["userId"].isin(user_ids)
        if before_timestamp is not None:
            mask &= chunk["timestamp"] < before_timestamp
        frames.append(chunk.loc[mask].copy())
    return pd.concat(frames, ignore_index=True)


def build_temporal_context(
    pairs: pd.DataFrame,
    ratings: pd.DataFrame,
    movie_ids: np.ndarray,
    config: PreprocessingConfig,
    *,
    period: str,
) -> TemporalEvaluationContext:
    """Use train history for validation and train+validation history for test.

    Models and movie catalogue stay frozen at the training cutoff. Test uses a
    single static ranking per pair, not online updates within the test period.
    """

    if period not in {"validation", "test"}:
        raise ValueError("period must be validation or test")
    user_ids = np.unique(pairs[["userA", "userB"]].to_numpy().ravel())
    split_ratings = add_temporal_split(ratings, config)
    history_splits = ("train",) if period == "validation" else ("train", "validation")
    seen = build_seen_item_sets(split_ratings, user_ids, available_splits=history_splits)
    positives = positive_events(split_ratings, period, user_ids)
    catalogue = set(np.asarray(movie_ids, dtype=np.int64).tolist())
    relevance = build_pair_relevant_item_sets(pairs, positives, catalogue, seen)
    positives_by_user = {
        int(user): set(group["movieId"].astype(int))
        for user, group in positives.groupby("userId")
    }
    total_targets = 0
    eligible_targets = 0
    two_sided_pairs = 0
    for row in pairs.itertuples(index=False):
        targets = [relevance[(int(row.pairId), int(user))] for user in (row.userA, row.userB)]
        two_sided_pairs += int(all(targets))
        for user, target in zip((row.userA, row.userB), targets):
            total_targets += len(positives_by_user.get(int(user), set()))
            eligible_targets += len(target)
    return TemporalEvaluationContext(
        pairs=pairs,
        seen=seen,
        relevance=relevance,
        catalog_size=len(catalogue),
        diagnostics={
            "period": period,
            "history_splits": list(history_splits),
            "pairs": len(pairs),
            "two_sided_evaluable_pairs": two_sided_pairs,
            "candidate_movies": len(catalogue),
            "pair_target_coverage_after_filtering": (
                eligible_targets / total_targets if total_targets else 0.0
            ),
            "member_count": len(user_ids),
            "pair_memberships": 2 * len(pairs),
        },
    )
