"""Held-out relevance diagnostics computed only after pair construction."""

from __future__ import annotations

import pandas as pd


def add_joint_test_counts(
    pairs: pd.DataFrame,
    test_positive_events: pd.DataFrame,
) -> pd.DataFrame:
    """Add observable common positive counts without changing pair selection."""

    positive_sets = {
        int(user_id): set(group["movieId"].astype(int))
        for user_id, group in test_positive_events.groupby("userId", sort=False)
    }
    result = pairs.copy()
    result["jointTestPositiveCount"] = [
        len(positive_sets.get(int(row.userA), set()) & positive_sets.get(int(row.userB), set()))
        for row in result.itertuples(index=False)
    ]
    return result
