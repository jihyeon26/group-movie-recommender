"""Full-catalogue pair scoring from exported user and movie embeddings."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .group_ranking import (
    normalize_member_scores,
    rank_group_candidates,
    rank_nash_group_candidates,
)


def fold_in_user_embedding(
    ratings: pd.DataFrame,
    movie_ids: np.ndarray,
    movie_embeddings: np.ndarray,
    *,
    positive_threshold: float = 4.0,
) -> np.ndarray:
    """Estimate a new user's vector from positively rated item embeddings.

    LightGCN is transductive, so a user absent from the training graph has no
    learned embedding. This deterministic fold-in uses the same positive-rating
    threshold as the graph and gives higher ratings slightly more weight.
    """

    required = {"movieId", "rating"}
    missing = required - set(ratings.columns)
    if missing:
        raise ValueError(f"Ratings are missing columns: {sorted(missing)}")
    if ratings["movieId"].duplicated().any():
        raise ValueError("A fold-in profile can contain only one rating per movieId")
    known_ids = np.asarray(movie_ids, dtype=np.int64)
    embeddings = np.asarray(movie_embeddings, dtype=float)
    if embeddings.ndim != 2 or len(known_ids) != len(embeddings):
        raise ValueError("Movie IDs and a two-dimensional embedding matrix must align")
    if len(known_ids) == 0 or len(np.unique(known_ids)) != len(known_ids):
        raise ValueError("Movie IDs must be non-empty and unique")

    positive = ratings.loc[
        pd.to_numeric(ratings["rating"], errors="coerce") >= positive_threshold,
        ["movieId", "rating"],
    ].copy()
    positive["movieId"] = pd.to_numeric(positive["movieId"], errors="coerce")
    positive = positive.dropna(subset=["movieId", "rating"])
    position_by_id = {int(movie_id): index for index, movie_id in enumerate(known_ids)}
    positive = positive.loc[positive["movieId"].astype(int).isin(position_by_id)]
    if positive.empty:
        raise ValueError("No positive ratings overlap the trained movie embeddings")

    positions = np.array(
        [position_by_id[int(movie_id)] for movie_id in positive["movieId"]],
        dtype=np.int64,
    )
    # At threshold 4.0, weights are 0.5, 1.0, and 1.5 for ratings 4, 4.5, and 5.
    weights = positive["rating"].to_numpy(dtype=float) - positive_threshold + 0.5
    return np.average(embeddings[positions], axis=0, weights=weights)


def score_folded_in_pair(
    ratings_a: pd.DataFrame,
    ratings_b: pd.DataFrame,
    movie_ids: np.ndarray,
    movie_embeddings: np.ndarray,
    *,
    positive_threshold: float = 4.0,
) -> pd.DataFrame:
    """Score all unseen model movies for two users folded into item space."""

    user_a = fold_in_user_embedding(
        ratings_a,
        movie_ids,
        movie_embeddings,
        positive_threshold=positive_threshold,
    )
    user_b = fold_in_user_embedding(
        ratings_b,
        movie_ids,
        movie_embeddings,
        positive_threshold=positive_threshold,
    )
    seen = set(ratings_a["movieId"].astype(int)) | set(ratings_b["movieId"].astype(int))
    model_movie_ids = np.asarray(movie_ids, dtype=np.int64)
    model_movie_embeddings = np.asarray(movie_embeddings, dtype=float)
    candidate_mask = ~np.isin(model_movie_ids, np.fromiter(seen, dtype=np.int64))
    if not candidate_mask.any():
        raise ValueError("No unseen movies remain in the embedding catalogue")
    candidates = model_movie_embeddings[candidate_mask]
    return pd.DataFrame(
        {
            "movieId": model_movie_ids[candidate_mask],
            "scoreA": candidates @ user_a,
            "scoreB": candidates @ user_b,
        }
    )


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


def recommend_pairs_from_score_matrix(
    pairs: pd.DataFrame,
    user_ids: np.ndarray,
    user_scores: np.ndarray,
    movie_ids: np.ndarray,
    seen_items: dict[int, set[int]],
    *,
    conflict_weight: float,
    k: int,
    aggregation: str = "linear",
) -> pd.DataFrame:
    """Rank pairs from precomputed full-catalogue scores such as ItemKNN."""

    users = np.asarray(user_ids, dtype=np.int64)
    movies = np.asarray(movie_ids, dtype=np.int64)
    scores = np.asarray(user_scores, dtype=float)
    if scores.shape != (len(users), len(movies)):
        raise ValueError("user_scores must have shape (number of users, number of movies)")
    if not np.isfinite(scores).all():
        raise ValueError("user_scores must be finite")
    if aggregation not in {"linear", "nash"}:
        raise ValueError("aggregation must be linear or nash")
    records = []
    catalogue = set(movies.tolist())
    for fallback_pair_id, row in enumerate(pairs.itertuples(index=False)):
        pair_id = int(getattr(row, "pairId", fallback_pair_id))
        member_ids = np.array([int(row.userA), int(row.userB)])
        positions = _lookup_positions(users, member_ids)
        available = catalogue - (
            seen_items.get(int(row.userA), set()) | seen_items.get(int(row.userB), set())
        )
        mask = np.isin(movies, np.fromiter(available, dtype=np.int64))
        raw = pd.DataFrame({
            "movieId": movies[mask],
            "scoreA": scores[positions[0], mask],
            "scoreB": scores[positions[1], mask],
        })
        normalized = normalize_member_scores(raw)
        if aggregation == "nash":
            ranked = rank_nash_group_candidates(normalized, k=k)
        else:
            ranked = rank_group_candidates(
                normalized, conflict_weight=conflict_weight, k=k,
            )
        ranked.insert(1, "pairId", pair_id)
        ranked.insert(2, "userA", int(row.userA))
        ranked.insert(3, "userB", int(row.userB))
        records.append(ranked)
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
