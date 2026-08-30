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
    """Sample positive edges and unseen negative movies for one BPR batch."""

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    positives_by_user = positive_movie_sets(graph)
    saturated_users = [
        user_index
        for user_index, movie_set in enumerate(positives_by_user)
        if len(movie_set) == graph.num_movies
    ]
    eligible_edge_positions = np.flatnonzero(
        ~np.isin(graph.positive_user_indices, saturated_users)
    )
    if not len(eligible_edge_positions):
        raise ValueError("No user has an unseen movie available for negative sampling")

    rng = np.random.default_rng(random_seed)
    sampled_positions = rng.choice(
        eligible_edge_positions,
        size=batch_size,
        replace=True,
    )
    user_indices = graph.positive_user_indices[sampled_positions]
    positive_movie_indices = graph.positive_movie_indices[sampled_positions]
    negative_movie_indices = rng.integers(
        0,
        graph.num_movies,
        size=batch_size,
        dtype=np.int64,
    )

    invalid = np.fromiter(
        (
            int(movie_index) in positives_by_user[int(user_index)]
            for user_index, movie_index in zip(user_indices, negative_movie_indices)
        ),
        dtype=bool,
        count=batch_size,
    )
    while invalid.any():
        negative_movie_indices[invalid] = rng.integers(
            0,
            graph.num_movies,
            size=int(invalid.sum()),
            dtype=np.int64,
        )
        invalid_positions = np.flatnonzero(invalid)
        invalid[invalid_positions] = np.fromiter(
            (
                int(negative_movie_indices[position])
                in positives_by_user[int(user_indices[position])]
                for position in invalid_positions
            ),
            dtype=bool,
            count=len(invalid_positions),
        )

    return pd.DataFrame(
        {
            "userIndex": user_indices,
            "positiveMovieIndex": positive_movie_indices,
            "negativeMovieIndex": negative_movie_indices,
        }
    )
