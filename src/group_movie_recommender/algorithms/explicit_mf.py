"""Biased matrix factorization trained on the full explicit rating scale."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn


@dataclass(frozen=True)
class ExplicitMFConfig:
    factors: int = 32
    epochs: int = 20
    learning_rate: float = 0.01
    regularization: float = 0.0001
    batch_size: int = 4096
    random_seed: int = 20_260_828
    patience: int = 4
    validation_interval: int = 1

    def __post_init__(self) -> None:
        integers = (
            self.factors, self.epochs, self.batch_size,
            self.patience, self.validation_interval,
        )
        if min(integers) <= 0:
            raise ValueError("integer explicit-MF settings must be positive")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(self.regularization) or self.regularization < 0:
            raise ValueError("regularization must be finite and non-negative")


@dataclass(frozen=True)
class ExplicitRatingData:
    user_ids: np.ndarray
    movie_ids: np.ndarray
    user_indices: np.ndarray
    movie_indices: np.ndarray
    ratings: np.ndarray


@dataclass(frozen=True)
class ExplicitMFResult:
    artifact: dict[str, np.ndarray]
    history: pd.DataFrame
    best_epoch: int
    stopped_epoch: int
    early_stopped: bool


class BiasedMatrixFactorization(nn.Module):
    """Global mean plus user/item biases and a latent dot product."""

    def __init__(
        self,
        num_users: int,
        num_movies: int,
        factors: int,
        global_mean: float,
        *,
        random_seed: int,
    ) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(random_seed)
            self.user_factors = nn.Embedding(num_users, factors)
            self.movie_factors = nn.Embedding(num_movies, factors)
            nn.init.normal_(self.user_factors.weight, std=0.05)
            nn.init.normal_(self.movie_factors.weight, std=0.05)
        self.user_bias = nn.Embedding(num_users, 1)
        self.movie_bias = nn.Embedding(num_movies, 1)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)
        self.register_buffer("global_mean", torch.tensor(float(global_mean)))

    def forward(self, users: torch.Tensor, movies: torch.Tensor) -> torch.Tensor:
        interaction = (
            self.user_factors(users) * self.movie_factors(movies)
        ).sum(dim=1)
        return (
            self.global_mean
            + self.user_bias(users).squeeze(1)
            + self.movie_bias(movies).squeeze(1)
            + interaction
        )


def build_explicit_rating_data(
    ratings: pd.DataFrame,
    user_ids: np.ndarray,
    movie_ids: np.ndarray,
) -> ExplicitRatingData:
    """Map known user/movie IDs to the fixed experiment indices."""

    required = {"userId", "movieId", "rating"}
    missing = required - set(ratings.columns)
    if missing:
        raise ValueError(f"Ratings are missing columns: {sorted(missing)}")
    users = np.asarray(user_ids, dtype=np.int64)
    movies = np.asarray(movie_ids, dtype=np.int64)
    if (
        not len(users) or not len(movies)
        or not np.all(users[:-1] < users[1:])
        or not np.all(movies[:-1] < movies[1:])
    ):
        raise ValueError("user_ids and movie_ids must be non-empty, sorted, and unique")
    frame = ratings.loc[:, ["userId", "movieId", "rating"]].copy()
    if frame.duplicated(["userId", "movieId"]).any():
        raise ValueError("Explicit training data contain duplicate user-movie ratings")
    frame = frame.loc[
        frame["userId"].isin(users) & frame["movieId"].isin(movies)
    ]
    values = frame["rating"].to_numpy(dtype=np.float32)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("At least one finite known rating is required")
    if (values < 0.5).any() or (values > 5.0).any():
        raise ValueError("MovieLens ratings must be between 0.5 and 5.0")
    user_indices = np.searchsorted(
        users, frame["userId"].to_numpy(dtype=np.int64)
    ).astype(np.int64)
    movie_indices = np.searchsorted(
        movies, frame["movieId"].to_numpy(dtype=np.int64)
    ).astype(np.int64)
    return ExplicitRatingData(
        user_ids=users,
        movie_ids=movies,
        user_indices=user_indices,
        movie_indices=movie_indices,
        ratings=values,
    )


def train_explicit_mf(
    data: ExplicitRatingData,
    config: ExplicitMFConfig,
    *,
    validation_callback: Callable[[dict[str, np.ndarray], int], tuple[float, float]]
    | None = None,
) -> ExplicitMFResult:
    """Train with squared error and restore the best recommendation checkpoint."""

    model = BiasedMatrixFactorization(
        len(data.user_ids), len(data.movie_ids), config.factors,
        float(data.ratings.mean()), random_seed=config.random_seed,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.regularization,
    )
    users = torch.from_numpy(data.user_indices)
    movies = torch.from_numpy(data.movie_indices)
    ratings = torch.from_numpy(data.ratings)
    generator = torch.Generator().manual_seed(config.random_seed)
    best_key: tuple[float, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale = 0
    stopped_epoch = config.epochs
    rows: list[dict] = []

    for epoch in range(1, config.epochs + 1):
        permutation = torch.randperm(len(ratings), generator=generator)
        squared_error = 0.0
        model.train()
        for start in range(0, len(ratings), config.batch_size):
            batch = permutation[start:start + config.batch_size]
            prediction = model(users[batch], movies[batch])
            loss = torch.mean((prediction - ratings[batch]) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            squared_error += float(loss.detach()) * len(batch)
        row: dict = {
            "epoch": epoch,
            "training_rmse": float(np.sqrt(squared_error / len(ratings))),
        }
        should_validate = (
            validation_callback is not None
            and (epoch % config.validation_interval == 0 or epoch == config.epochs)
        )
        if should_validate:
            artifact = explicit_mf_artifact(model, data.user_ids, data.movie_ids)
            key = tuple(float(value) for value in validation_callback(artifact, epoch))
            if len(key) != 2 or not np.isfinite(key).all():
                raise ValueError("Validation must return two finite scores")
            improved = best_key is None or key > best_key
            row.update(
                validation_primary=key[0],
                validation_secondary=key[1],
                checkpoint_improved=improved,
            )
            if improved:
                best_key = key
                best_state = {
                    name: value.detach().clone()
                    for name, value in model.state_dict().items()
                }
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
        rows.append(row)
        if validation_callback is not None and stale >= config.patience:
            stopped_epoch = epoch
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = stopped_epoch
    artifact = explicit_mf_artifact(model, data.user_ids, data.movie_ids)
    return ExplicitMFResult(
        artifact=artifact,
        history=pd.DataFrame(rows),
        best_epoch=best_epoch,
        stopped_epoch=stopped_epoch,
        early_stopped=stopped_epoch < config.epochs,
    )


def explicit_mf_artifact(
    model: BiasedMatrixFactorization,
    user_ids: np.ndarray,
    movie_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    """Export all parameters needed for deterministic full-catalogue scoring."""

    with torch.no_grad():
        return {
            "user_ids": np.asarray(user_ids, dtype=np.int64),
            "movie_ids": np.asarray(movie_ids, dtype=np.int64),
            "user_factors": model.user_factors.weight.cpu().numpy().copy(),
            "movie_factors": model.movie_factors.weight.cpu().numpy().copy(),
            "user_bias": model.user_bias.weight.squeeze(1).cpu().numpy().copy(),
            "movie_bias": model.movie_bias.weight.squeeze(1).cpu().numpy().copy(),
            "global_mean": np.array(float(model.global_mean)),
        }


def score_explicit_mf_users(
    artifact: dict[str, np.ndarray],
    requested_user_ids: np.ndarray,
) -> np.ndarray:
    """Predict ratings for requested trained users against the full catalogue."""

    known_users = np.asarray(artifact["user_ids"], dtype=np.int64)
    requested = np.asarray(requested_user_ids, dtype=np.int64)
    positions = np.searchsorted(known_users, requested)
    valid = positions < len(known_users)
    valid[valid] &= known_users[positions[valid]] == requested[valid]
    if not valid.all():
        raise KeyError(f"Users missing from explicit-MF artifact: {requested[~valid].tolist()}")
    user_factors = np.asarray(artifact["user_factors"], dtype=np.float32)[positions]
    movie_factors = np.asarray(artifact["movie_factors"], dtype=np.float32)
    user_bias = np.asarray(artifact["user_bias"], dtype=np.float32)[positions]
    movie_bias = np.asarray(artifact["movie_bias"], dtype=np.float32)
    global_mean = float(np.asarray(artifact["global_mean"]))
    return (
        global_mean
        + user_bias[:, None]
        + movie_bias[None, :]
        + user_factors @ movie_factors.T
    )
