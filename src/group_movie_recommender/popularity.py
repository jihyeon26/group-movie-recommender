"""A transparent non-personalized popularity recommender."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PopularityRecommender:
    """Rank warm movies by their number of positive training interactions."""

    ranked_movie_ids: np.ndarray
    scores: dict[int, float]

    @classmethod
    def from_warm_catalog(cls, warm_catalog: pd.DataFrame) -> "PopularityRecommender":
        """Fit the baseline from a warm catalogue produced with training data only."""

        required = {"movieId", "trainPositiveCount"}
        missing = required - set(warm_catalog.columns)
        if missing:
            raise ValueError(f"Warm catalogue is missing columns: {sorted(missing)}")

        ranked = warm_catalog.sort_values(
            ["trainPositiveCount", "movieId"],
            ascending=[False, True],
        )
        movie_ids = ranked["movieId"].to_numpy(dtype=np.int32)
        scores = dict(
            zip(
                movie_ids.astype(int),
                ranked["trainPositiveCount"].astype(float),
            )
        )
        return cls(ranked_movie_ids=movie_ids, scores=scores)

    def recommend(
        self,
        seen_by_user_a: set[int],
        seen_by_user_b: set[int],
        *,
        k: int = 10,
    ) -> list[tuple[int, float]]:
        """Return the most popular warm movies unseen by both group members."""

        if k <= 0:
            raise ValueError("k must be a positive integer")

        unavailable = seen_by_user_a | seen_by_user_b
        recommendations: list[tuple[int, float]] = []
        for movie_id in self.ranked_movie_ids:
            item = int(movie_id)
            if item in unavailable:
                continue
            recommendations.append((item, self.scores[item]))
            if len(recommendations) == k:
                break
        return recommendations


def recommend_pairs_by_popularity(
    pairs: pd.DataFrame,
    warm_catalog: pd.DataFrame,
    seen_items: dict[int, set[int]],
    *,
    k: int = 10,
) -> pd.DataFrame:
    """Create one shared popularity-ranked list for every two-user pair."""

    required = {"userA", "userB"}
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"Pair table is missing columns: {sorted(missing)}")

    model = PopularityRecommender.from_warm_catalog(warm_catalog)
    records: list[dict[str, int | float]] = []
    for pair_id, pair in pairs.reset_index(drop=True).iterrows():
        user_a = int(pair["userA"])
        user_b = int(pair["userB"])
        ranked_items = model.recommend(
            seen_items.get(user_a, set()),
            seen_items.get(user_b, set()),
            k=k,
        )
        for rank, (movie_id, score) in enumerate(ranked_items, start=1):
            records.append(
                {
                    "pairId": int(pair_id),
                    "userA": user_a,
                    "userB": user_b,
                    "rank": rank,
                    "movieId": movie_id,
                    "score": score,
                }
            )
    return pd.DataFrame.from_records(
        records,
        columns=["pairId", "userA", "userB", "rank", "movieId", "score"],
    )
