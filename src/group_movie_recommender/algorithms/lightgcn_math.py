"""Framework-independent LightGCN propagation, scoring, and BPR loss."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LightGCNPropagation:
    """Embeddings from every propagation depth and their LightGCN average."""

    layer_embeddings: tuple[np.ndarray, ...]
    final_embeddings: np.ndarray


def validate_edge_index(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    """Validate and return a two-row integer edge index."""

    edges = np.asarray(edge_index)
    if num_nodes <= 0:
        raise ValueError("num_nodes must be a positive integer")
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, number_of_edges)")
    if edges.shape[1] == 0:
        raise ValueError("edge_index must contain at least one edge")
    if not np.issubdtype(edges.dtype, np.integer):
        raise ValueError("edge_index must contain integer node indices")
    if edges.min() < 0 or edges.max() >= num_nodes:
        raise ValueError("edge_index contains a node outside the valid range")
    return edges.astype(np.int64, copy=False)


def normalized_edge_weights(
    edge_index: np.ndarray,
    *,
    num_nodes: int,
) -> np.ndarray:
    """Compute symmetric LightGCN weights 1 / sqrt(deg(source) deg(target))."""

    edges = validate_edge_index(edge_index, num_nodes)
    source, target = edges
    degree = np.bincount(source, minlength=num_nodes).astype(np.float64)
    if (degree == 0).any():
        isolated = np.flatnonzero(degree == 0).tolist()
        raise ValueError(f"Every node must have an outgoing edge; isolated nodes: {isolated}")
    return 1.0 / np.sqrt(degree[source] * degree[target])


def propagate_once(
    node_embeddings: np.ndarray,
    edge_index: np.ndarray,
) -> np.ndarray:
    """Aggregate normalized neighbor embeddings for one LightGCN layer."""

    embeddings = np.asarray(node_embeddings, dtype=np.float64)
    if embeddings.ndim != 2 or embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
        raise ValueError("node_embeddings must have shape (num_nodes, embedding_dim)")

    edges = validate_edge_index(edge_index, len(embeddings))
    source, target = edges
    weights = normalized_edge_weights(edges, num_nodes=len(embeddings))
    propagated = np.zeros_like(embeddings)
    np.add.at(
        propagated,
        target,
        embeddings[source] * weights[:, None],
    )
    return propagated


def lightgcn_propagate(
    initial_embeddings: np.ndarray,
    edge_index: np.ndarray,
    *,
    num_layers: int,
) -> LightGCNPropagation:
    """Propagate embeddings and average layer zero through the final layer."""

    if num_layers < 0:
        raise ValueError("num_layers cannot be negative")

    initial = np.asarray(initial_embeddings, dtype=np.float64)
    if initial.ndim != 2 or initial.shape[0] == 0 or initial.shape[1] == 0:
        raise ValueError("initial_embeddings must have shape (num_nodes, embedding_dim)")
    validate_edge_index(edge_index, len(initial))

    layers = [initial.copy()]
    for _ in range(num_layers):
        layers.append(propagate_once(layers[-1], edge_index))
    stacked = np.stack(layers, axis=0)
    return LightGCNPropagation(
        layer_embeddings=tuple(layers),
        final_embeddings=stacked.mean(axis=0),
    )


def split_user_movie_embeddings(
    node_embeddings: np.ndarray,
    *,
    num_users: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split global node embeddings into users and local-indexed movies."""

    embeddings = np.asarray(node_embeddings)
    if embeddings.ndim != 2:
        raise ValueError("node_embeddings must be a two-dimensional array")
    if not 0 < num_users < len(embeddings):
        raise ValueError("num_users must leave at least one user and one movie")
    return embeddings[:num_users], embeddings[num_users:]


def score_user_movie_pairs(
    user_embeddings: np.ndarray,
    movie_embeddings: np.ndarray,
    user_indices: np.ndarray,
    movie_indices: np.ndarray,
) -> np.ndarray:
    """Score aligned user--movie pairs with an embedding dot product."""

    users = np.asarray(user_indices, dtype=np.int64)
    movies = np.asarray(movie_indices, dtype=np.int64)
    if users.shape != movies.shape:
        raise ValueError("user_indices and movie_indices must have the same shape")
    return np.einsum(
        "ij,ij->i",
        np.asarray(user_embeddings)[users],
        np.asarray(movie_embeddings)[movies],
    )


def bpr_loss(positive_scores: np.ndarray, negative_scores: np.ndarray) -> float:
    """Return the stable mean loss -log(sigmoid(positive - negative))."""

    positive = np.asarray(positive_scores, dtype=np.float64)
    negative = np.asarray(negative_scores, dtype=np.float64)
    if positive.shape != negative.shape:
        raise ValueError("positive_scores and negative_scores must have the same shape")
    if positive.size == 0:
        raise ValueError("At least one score pair is required")
    return float(np.logaddexp(0.0, negative - positive).mean())
