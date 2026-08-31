"""Confidence-weighted implicit ALS on the indexed positive interaction graph."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .graph_data import BipartiteGraphData


@dataclass(frozen=True)
class ImplicitALSConfig:
    """Hyperparameters for binary-preference implicit ALS."""

    factors: int = 32
    iterations: int = 15
    regularization: float = 0.1
    alpha: float = 20.0
    batch_size: int = 256
    random_seed: int = 20_260_828
    patience: int = 4
    validation_interval: int = 1

    def __post_init__(self) -> None:
        if min(self.factors, self.iterations, self.batch_size, self.patience,
               self.validation_interval) <= 0:
            raise ValueError("integer ALS settings must be positive")
        if not np.isfinite(self.regularization) or self.regularization <= 0:
            raise ValueError("regularization must be finite and positive")
        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be finite and positive")


@dataclass(frozen=True)
class ImplicitALSResult:
    user_factors: np.ndarray
    movie_factors: np.ndarray
    history: pd.DataFrame
    best_iteration: int
    stopped_iteration: int
    early_stopped: bool


def train_implicit_als(
    graph: BipartiteGraphData,
    config: ImplicitALSConfig,
    *,
    validation_callback: Callable[[np.ndarray, np.ndarray, int], tuple[float, float]] | None = None,
) -> ImplicitALSResult:
    """Alternate exact least-squares updates and restore the best validation state.

    Every observed positive has preference one and confidence ``1 + alpha``;
    unobserved entries have preference zero and confidence one. Unobserved does
    not mean a confirmed dislike, so alpha is always selected on validation.
    """

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.random_seed)
        users = torch.randn(graph.num_users, config.factors, dtype=torch.float32) * 0.01
        movies = torch.randn(graph.num_movies, config.factors, dtype=torch.float32) * 0.01

    best_key: tuple[float, float] | None = None
    best_users: torch.Tensor | None = None
    best_movies: torch.Tensor | None = None
    best_iteration = 0
    stale = 0
    rows: list[dict] = []
    stopped_iteration = config.iterations

    for iteration in range(1, config.iterations + 1):
        users = _least_squares_update(
            movies, graph.positive_user_indices, graph.positive_movie_indices,
            graph.num_users, config,
        )
        movies = _least_squares_update(
            users, graph.positive_movie_indices, graph.positive_user_indices,
            graph.num_movies, config,
        )
        row: dict = {"iteration": iteration}
        should_validate = (
            validation_callback is not None
            and (iteration % config.validation_interval == 0 or iteration == config.iterations)
        )
        if should_validate:
            scores = tuple(float(value) for value in validation_callback(
                users.numpy(), movies.numpy(), iteration
            ))
            if len(scores) != 2 or not np.isfinite(scores).all():
                raise ValueError("Validation must return two finite scores")
            improved = best_key is None or scores > best_key
            row.update(
                validation_primary=scores[0],
                validation_secondary=scores[1],
                checkpoint_improved=improved,
            )
            if improved:
                best_key = scores
                best_users = users.clone()
                best_movies = movies.clone()
                best_iteration = iteration
                stale = 0
            else:
                stale += 1
        rows.append(row)
        if validation_callback is not None and stale >= config.patience:
            stopped_iteration = iteration
            break

    if best_users is not None and best_movies is not None:
        users, movies = best_users, best_movies
    else:
        best_iteration = stopped_iteration
    return ImplicitALSResult(
        user_factors=users.numpy(),
        movie_factors=movies.numpy(),
        history=pd.DataFrame(rows),
        best_iteration=best_iteration,
        stopped_iteration=stopped_iteration,
        early_stopped=stopped_iteration < config.iterations,
    )


def recalculate_user_factor(
    movie_factors: np.ndarray,
    positive_movie_indices: np.ndarray,
    config: ImplicitALSConfig,
) -> np.ndarray:
    """Fold a new user into a frozen ALS item space from positive movie indices."""

    movies = torch.as_tensor(movie_factors, dtype=torch.float32)
    positive = np.unique(np.asarray(positive_movie_indices, dtype=np.int64))
    if not len(positive) or positive.min() < 0 or positive.max() >= len(movies):
        raise ValueError("At least one valid positive movie index is required")
    selected = movies[torch.from_numpy(positive)]
    gram = movies.T @ movies
    lhs = gram + config.alpha * (selected.T @ selected)
    lhs += config.regularization * torch.eye(config.factors)
    rhs = (1.0 + config.alpha) * selected.sum(dim=0)
    return torch.linalg.solve(lhs, rhs).numpy()


def _least_squares_update(
    fixed_factors: torch.Tensor,
    row_indices: np.ndarray,
    column_indices: np.ndarray,
    num_rows: int,
    config: ImplicitALSConfig,
) -> torch.Tensor:
    """Solve independent normal equations in batches without a dense user-item matrix."""

    order = np.argsort(row_indices, kind="stable")
    sorted_columns = torch.from_numpy(np.asarray(column_indices[order], dtype=np.int64))
    counts_np = np.bincount(row_indices, minlength=num_rows).astype(np.int64)
    if (counts_np == 0).any():
        raise ValueError("ALS cannot update a graph row without a positive interaction")
    starts_np = np.concatenate(([0], np.cumsum(counts_np[:-1]))).astype(np.int64)
    result = torch.empty(num_rows, config.factors, dtype=torch.float32)
    gram = fixed_factors.T @ fixed_factors
    identity = torch.eye(config.factors, dtype=torch.float32)
    for first in range(0, num_rows, config.batch_size):
        last = min(first + config.batch_size, num_rows)
        lengths = torch.from_numpy(counts_np[first:last])
        starts = torch.from_numpy(starts_np[first:last])
        width = int(lengths.max().item())
        offsets = torch.arange(width).unsqueeze(0)
        valid = offsets < lengths.unsqueeze(1)
        positions = starts.unsqueeze(1) + offsets
        positions = positions.clamp(max=len(sorted_columns) - 1)
        neighbors = fixed_factors[sorted_columns[positions]] * valid.unsqueeze(2)
        covariance = torch.bmm(neighbors.transpose(1, 2), neighbors)
        lhs = gram.unsqueeze(0) + config.alpha * covariance
        lhs = lhs + config.regularization * identity.unsqueeze(0)
        rhs = (1.0 + config.alpha) * neighbors.sum(dim=1)
        result[first:last] = torch.linalg.solve(lhs, rhs.unsqueeze(2)).squeeze(2)
    return result
