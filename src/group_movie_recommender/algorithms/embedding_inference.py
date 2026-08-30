"""Full-catalogue pair scoring from exported user and movie embeddings."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .group_ranking import normalize_member_scores, rank_group_candidates


def recommend_pairs_from_embeddings(
    pairs: pd.DataFrame,
    user_ids: np.ndarray,
    user_embeddings: np.ndarray,
    movie_ids: np.ndarray,
    movie_embeddings: np.ndarray,
    seen_items: dict[int, set[int]],
    *,
    conflict_weight: float,
    k: int,
) -> pd.DataFrame:
    """Score graph movies, remove each pair's seen union, and return shared Top-k."""

    if len(user_ids) != len(user_embeddings) or len(movie_ids) != len(movie_embeddings):
        raise ValueError("ID arrays and embedding matrices must have matching row counts")
    if user_embeddings.ndim != 2 or movie_embeddings.ndim != 2:
        raise ValueError("Embedding arrays must be two-dimensional")
    if user_embeddings.shape[1] != movie_embeddings.shape[1]:
        raise ValueError("User and movie embedding dimensions must match")

    records: list[pd.DataFrame] = []
    catalogue = set(np.asarray(movie_ids, dtype=np.int64).tolist())
    for fallback_pair_id, row in enumerate(pairs.itertuples(index=False)):
        pair_id = int(getattr(row, "pairId", fallback_pair_id))
        user_a = int(row.userA)
        user_b = int(row.userB)
        positions = _lookup_positions(user_ids, np.array([user_a, user_b]))
        available = catalogue - (seen_items.get(user_a, set()) | seen_items.get(user_b, set()))
        mask = np.isin(movie_ids, np.fromiter(available, dtype=np.int64))
        candidate_movie_ids = np.asarray(movie_ids)[mask]
        candidate_embeddings = np.asarray(movie_embeddings)[mask]
        if not len(candidate_movie_ids):
            continue
        raw_scores = pd.DataFrame(
            {
                "movieId": candidate_movie_ids,
                "scoreA": candidate_embeddings @ user_embeddings[positions[0]],
                "scoreB": candidate_embeddings @ user_embeddings[positions[1]],
            }
        )
        normalized = normalize_member_scores(raw_scores)
        ranked = rank_group_candidates(
            normalized,
            conflict_weight=conflict_weight,
            k=k,
        )
        ranked.insert(1, "pairId", pair_id)
        ranked.insert(2, "userA", user_a)
        ranked.insert(3, "userB", user_b)
        records.append(ranked)
    if not records:
        return pd.DataFrame(columns=["rank", "pairId", "userA", "userB", "movieId"])
    return pd.concat(records, ignore_index=True)


def _lookup_positions(known_ids: np.ndarray, requested_ids: np.ndarray) -> np.ndarray:
    known = np.asarray(known_ids, dtype=np.int64)
    requested = np.asarray(requested_ids, dtype=np.int64)
    positions = np.searchsorted(known, requested)
    valid = positions < len(known)
    valid[valid] &= known[positions[valid]] == requested[valid]
    if not valid.all():
        raise KeyError(f"Users missing from embedding artifact: {requested[~valid].tolist()}")
    return positions
