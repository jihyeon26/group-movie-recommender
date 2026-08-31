"""Interpretable item-neighborhood collaborative filtering for implicit feedback."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from .graph_data import BipartiteGraphData, positive_movie_sets


@dataclass(frozen=True)
class ItemKNNModel:
    movie_ids: np.ndarray
    neighbor_indices: np.ndarray
    neighbor_scores: np.ndarray
    shrinkage: float

    @property
    def max_neighbors(self) -> int:
        return int(self.neighbor_indices.shape[1])


def fit_item_knn(
    graph: BipartiteGraphData,
    *,
    max_neighbors: int = 100,
    shrinkage: float = 0.0,
) -> ItemKNNModel:
    """Keep the strongest cosine co-occurrence neighbors for every movie."""

    if max_neighbors <= 0 or not np.isfinite(shrinkage) or shrinkage < 0:
        raise ValueError("max_neighbors must be positive and shrinkage non-negative")
    histories = positive_movie_sets(graph)
    item_users: list[list[int]] = [[] for _ in range(graph.num_movies)]
    for user, history in enumerate(histories):
        for movie in history:
            item_users[movie].append(user)
    degrees = np.array([len(users) for users in item_users], dtype=np.float64)
    width = min(max_neighbors, max(1, graph.num_movies - 1))
    neighbor_indices = np.full((graph.num_movies, width), -1, dtype=np.int32)
    neighbor_scores = np.zeros((graph.num_movies, width), dtype=np.float32)
    for source, users in enumerate(item_users):
        counts: Counter[int] = Counter()
        for user in users:
            counts.update(histories[user])
        counts.pop(source, None)
        similarities = [
            (candidate, count / (np.sqrt(degrees[source] * degrees[candidate]) + shrinkage))
            for candidate, count in counts.items()
        ]
        similarities.sort(key=lambda value: (-value[1], value[0]))
        selected = similarities[:width]
        if selected:
            neighbor_indices[source, :len(selected)] = [value[0] for value in selected]
            neighbor_scores[source, :len(selected)] = [value[1] for value in selected]
    return ItemKNNModel(
        movie_ids=graph.movie_ids.copy(),
        neighbor_indices=neighbor_indices,
        neighbor_scores=neighbor_scores,
        shrinkage=float(shrinkage),
    )


def score_item_knn_users(
    model: ItemKNNModel,
    graph: BipartiteGraphData,
    user_ids: np.ndarray,
    *,
    neighbors: int,
) -> np.ndarray:
    """Sum similarities from each user's positive training movies."""

    if not 0 < neighbors <= model.max_neighbors:
        raise ValueError("neighbors must be within the fitted model width")
    if not np.array_equal(model.movie_ids, graph.movie_ids):
        raise ValueError("ItemKNN model and graph movie IDs differ")
    histories = positive_movie_sets(graph)
    user_positions = graph.user_indices(np.asarray(user_ids, dtype=np.int64))
    result = np.zeros((len(user_positions), graph.num_movies), dtype=np.float32)
    for row, position in enumerate(user_positions):
        for source in histories[int(position)]:
            candidate_ids = model.neighbor_indices[source, :neighbors]
            values = model.neighbor_scores[source, :neighbors]
            valid = candidate_ids >= 0
            np.add.at(result[row], candidate_ids[valid], values[valid])
    return result


def score_item_knn_profile(
    model: ItemKNNModel,
    positive_movie_ids: np.ndarray,
    *,
    neighbors: int,
) -> np.ndarray:
    """Score the catalogue from an external user's positive movie IDs."""

    if not 0 < neighbors <= model.max_neighbors:
        raise ValueError("neighbors must be within the fitted model width")
    requested = np.unique(np.asarray(positive_movie_ids, dtype=np.int64))
    positions = np.searchsorted(model.movie_ids, requested)
    valid = positions < len(model.movie_ids)
    valid[valid] &= model.movie_ids[positions[valid]] == requested[valid]
    positions = positions[valid]
    if not len(positions):
        raise ValueError("No positive ratings overlap the ItemKNN catalogue")

    result = np.zeros(len(model.movie_ids), dtype=np.float32)
    for source in positions:
        candidate_ids = model.neighbor_indices[source, :neighbors]
        values = model.neighbor_scores[source, :neighbors]
        present = candidate_ids >= 0
        np.add.at(result, candidate_ids[present], values[present])
    return result


def save_item_knn(model: ItemKNNModel, path) -> None:
    np.savez_compressed(
        path,
        movie_ids=model.movie_ids,
        neighbor_indices=model.neighbor_indices,
        neighbor_scores=model.neighbor_scores,
        shrinkage=np.array(model.shrinkage),
    )


def load_item_knn(path) -> ItemKNNModel:
    with np.load(path, allow_pickle=False) as archive:
        return ItemKNNModel(
            movie_ids=archive["movie_ids"],
            neighbor_indices=archive["neighbor_indices"],
            neighbor_scores=archive["neighbor_scores"],
            shrinkage=float(archive["shrinkage"]),
        )
