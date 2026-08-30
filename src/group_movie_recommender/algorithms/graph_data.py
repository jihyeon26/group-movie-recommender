"""Indexing, bipartite edges, and BPR sampling for graph recommenders."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BipartiteGraphData:
    """Compact indexed representation of positive user--movie interactions."""

    user_ids: np.ndarray
    movie_ids: np.ndarray
    positive_user_indices: np.ndarray
    positive_movie_indices: np.ndarray
    edge_index: np.ndarray

    @property
    def num_users(self) -> int:
        """Return the number of user nodes."""

        return int(len(self.user_ids))

    @property
    def num_movies(self) -> int:
        """Return the number of movie nodes."""

        return int(len(self.movie_ids))

    @property
    def num_nodes(self) -> int:
        """Return the total number of user and movie nodes."""

        return self.num_users + self.num_movies

    @property
    def num_positive_edges(self) -> int:
        """Return the number of unique user--movie interactions."""

        return int(len(self.positive_user_indices))

    def original_user_ids(self, user_indices: np.ndarray) -> np.ndarray:
        """Decode contiguous user indices to MovieLens user IDs."""

        return self.user_ids[np.asarray(user_indices, dtype=np.int64)]

    def original_movie_ids(self, movie_indices: np.ndarray) -> np.ndarray:
        """Decode local movie indices to MovieLens movie IDs."""

        return self.movie_ids[np.asarray(movie_indices, dtype=np.int64)]

    def user_indices(self, original_user_ids: np.ndarray) -> np.ndarray:
        """Encode MovieLens user IDs and reject users outside this graph."""

        return _encode_sorted_ids(self.user_ids, original_user_ids, name="userId")

    def movie_indices(self, original_movie_ids: np.ndarray) -> np.ndarray:
        """Encode MovieLens movie IDs and reject movies outside this graph."""

        return _encode_sorted_ids(self.movie_ids, original_movie_ids, name="movieId")


def _encode_sorted_ids(
    known_ids: np.ndarray,
    requested_ids: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    requested = np.asarray(requested_ids, dtype=np.int64)
    positions = np.searchsorted(known_ids, requested)
    valid = positions < len(known_ids)
    valid[valid] &= known_ids[positions[valid]] == requested[valid]
    if not valid.all():
        raise KeyError(f"Unknown {name} values: {requested[~valid].tolist()}")
    return positions.astype(np.int64)


def build_bipartite_graph(train_edges: pd.DataFrame) -> BipartiteGraphData:
    """Map positive train edges to contiguous user and movie node indices."""

    required = {"userId", "movieId"}
    missing = required - set(train_edges.columns)
    if missing:
        raise ValueError(f"Training edges are missing columns: {sorted(missing)}")
    if train_edges.empty:
        raise ValueError("At least one positive training edge is required")
    if train_edges[["userId", "movieId"]].isna().any().any():
        raise ValueError("Training edges cannot contain missing IDs")

    unique_edges = (
        train_edges.loc[:, ["userId", "movieId"]]
        .drop_duplicates()
        .sort_values(["userId", "movieId"])
        .reset_index(drop=True)
    )
    user_ids = np.sort(unique_edges["userId"].unique()).astype(np.int64)
    movie_ids = np.sort(unique_edges["movieId"].unique()).astype(np.int64)
    user_indices = np.searchsorted(
        user_ids,
        unique_edges["userId"].to_numpy(dtype=np.int64),
    )
    movie_indices = np.searchsorted(
        movie_ids,
        unique_edges["movieId"].to_numpy(dtype=np.int64),
    )

    movie_node_indices = len(user_ids) + movie_indices
    forward_edges = np.vstack([user_indices, movie_node_indices])
    reverse_edges = forward_edges[::-1]
    edge_index = np.hstack([forward_edges, reverse_edges]).astype(np.int64)

    return BipartiteGraphData(
        user_ids=user_ids,
        movie_ids=movie_ids,
        positive_user_indices=user_indices.astype(np.int64),
        positive_movie_indices=movie_indices.astype(np.int64),
        edge_index=edge_index,
    )


def positive_movie_sets(graph: BipartiteGraphData) -> tuple[frozenset[int], ...]:
    """Return each indexed user's positive local movie indices."""

    items: list[set[int]] = [set() for _ in range(graph.num_users)]
    for user_index, movie_index in zip(
        graph.positive_user_indices,
        graph.positive_movie_indices,
    ):
        items[int(user_index)].add(int(movie_index))
    return tuple(frozenset(user_items) for user_items in items)


def sample_bpr_batch(
    graph: BipartiteGraphData,
    *,
    batch_size: int,
    random_seed: int,
) -> pd.DataFrame:
    """Draw one reproducible batch; reuse BPRBatchSampler inside training loops."""

    return BPRBatchSampler(graph, random_seed=random_seed).sample(batch_size=batch_size)


class BPRBatchSampler:
    """Cache positive sets and advance one RNG across successive training batches.

    Negatives are absent from the positive training graph, not necessarily unrated:
    low ratings can also be sampled. Validation/test labels are never consulted.
    """

    def __init__(self, graph: BipartiteGraphData, *, random_seed: int) -> None:
        self.graph = graph
        self.rng = np.random.default_rng(random_seed)
        self.positives_by_user = positive_movie_sets(graph)
        saturated_users = [
            user for user, items in enumerate(self.positives_by_user)
            if len(items) == graph.num_movies
        ]
        self.eligible_edge_positions = np.flatnonzero(
            ~np.isin(graph.positive_user_indices, saturated_users)
        )
        if not len(self.eligible_edge_positions):
            raise ValueError("No user has a non-positive movie available for negative sampling")

    def sample(self, *, batch_size: int) -> pd.DataFrame:
        """Sample positive edges uniformly with replacement and exclude known positives."""

        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        positions = self.rng.choice(self.eligible_edge_positions, size=batch_size, replace=True)
        users = self.graph.positive_user_indices[positions]
        positives = self.graph.positive_movie_indices[positions]
        negatives = self.rng.integers(0, self.graph.num_movies, size=batch_size, dtype=np.int64)

        # Bound rejection attempts so near-saturated users cannot stall a batch.
        for _ in range(32):
            invalid = np.fromiter(
                (int(item) in self.positives_by_user[int(user)] for user, item in zip(users, negatives)),
                dtype=bool,
                count=batch_size,
            )
            if not invalid.any():
                break
            negatives[invalid] = self.rng.integers(
                0, self.graph.num_movies, size=int(invalid.sum()), dtype=np.int64,
            )

        for position, (user, item) in enumerate(zip(users, negatives)):
            known = self.positives_by_user[int(user)]
            if int(item) in known:
                available = np.setdiff1d(np.arange(self.graph.num_movies), list(known))
                negatives[position] = self.rng.choice(available)

        return pd.DataFrame({
            "userIndex": users,
            "positiveMovieIndex": positives,
            "negativeMovieIndex": negatives,
        })
