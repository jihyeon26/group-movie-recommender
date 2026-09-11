"""Inspectable mini-batch training on small graphs, before MovieLens scaling."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch

from ..algorithms.graph_data import BPRBatchSampler, BipartiteGraphData, build_bipartite_graph
from ..algorithms.lightgcn import LightGCN


@dataclass(frozen=True)
class LightGCNTrainingConfig:
    """CPU toy-training settings; these are not tuned MovieLens hyperparameters."""

    embedding_dim: int = 16
    num_layers: int = 2
    steps: int = 100
    batch_size: int = 64
    learning_rate: float = 0.05
    l2_weight: float = 0.0001
    random_seed: int = 7

    def __post_init__(self) -> None:
        if min(self.embedding_dim, self.steps, self.batch_size) <= 0 or self.num_layers < 0:
            raise ValueError("Dimensions, steps, and batch size must be positive; layers >= 0")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(self.l2_weight) or self.l2_weight < 0:
            raise ValueError("l2_weight must be finite and non-negative")


def build_toy_training_graph() -> BipartiteGraphData:
    """Create four synthetic users and five movies for a quick CPU learning check."""

    return build_bipartite_graph(pd.DataFrame({
        "userId": [10, 10, 20, 20, 30, 30, 40, 40],
        "movieId": [100, 200, 200, 300, 100, 400, 300, 500],
    }))


def batch_to_tensors(batch: pd.DataFrame) -> tuple[torch.Tensor, ...]:
    """Convert local user/movie indices to CPU tensors without changing their meaning."""

    return tuple(
        torch.as_tensor(batch[column].to_numpy(dtype=np.int64), dtype=torch.long)
        for column in ("userIndex", "positiveMovieIndex", "negativeMovieIndex")
    )


def train_lightgcn(
    graph: BipartiteGraphData,
    config: LightGCNTrainingConfig,
    *,
    validation_callback: Callable[[LightGCN, int], tuple[float, float]] | None = None,
    validation_interval: int = 100,
    patience: int = 4,
    selection_smoothing: int = 1,
) -> tuple[LightGCN, pd.DataFrame]:
    """Train on sampled positives; monitor a fixed training diagnostic batch.

    The monitor is NOT a validation/test set or a recommendation metric. This CPU
    reference propagates over the complete graph each update and is not yet a
    throughput-tuned MovieLens 32M trainer. An optional callback returns a
    validation (primary, tie-breaker) score; larger is better. When supplied,
    restore the best checkpoint, not the last update. Exact ties retain the
    earlier checkpoint. Selection and stopping use a trailing mean over the last
    selection_smoothing validation estimates, so a noisy primary metric cannot
    end training on a single sampling spike. History attrs contain
    checkpoint-selection metadata.
    """

    if validation_interval <= 0 or patience <= 0:
        raise ValueError("validation_interval and patience must be positive")
    if selection_smoothing <= 0:
        raise ValueError("selection_smoothing must be a positive integer")

    model = LightGCN(
        graph,
        embedding_dim=config.embedding_dim,
        num_layers=config.num_layers,
        random_seed=config.random_seed,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    sampler = BPRBatchSampler(graph, random_seed=config.random_seed + 1)
    monitor_sampler = BPRBatchSampler(graph, random_seed=config.random_seed + 2)
    monitor = batch_to_tensors(monitor_sampler.sample(batch_size=config.batch_size))

    def fixed_batch_loss() -> float:
        with torch.no_grad():
            return float(model.bpr_objective(*monitor, l2_weight=0.0)["bpr_loss"].item())

    rows = [{"step": 0, "fixed_training_bpr_loss": fixed_batch_loss()}]
    best_key: tuple[float, float] | None = None
    best_weights: torch.Tensor | None = None
    best_step = 0
    stale_checks = 0
    recent_scores: list[tuple[float, float]] = []

    def validate(step: int, row: dict) -> bool:
        nonlocal best_key, best_weights, best_step, stale_checks
        if validation_callback is None:
            return False
        model.eval()
        with torch.no_grad():
            scores = tuple(float(value) for value in validation_callback(model, step))
        model.train()
        if len(scores) != 2 or not np.isfinite(scores).all():
            raise ValueError("Validation must return two finite scores")

        # Selecting on a single noisy validation estimate picks sampling spikes and
        # stops training early. Compare a trailing mean instead; a window of one
        # reproduces the original single-estimate rule.
        recent_scores.append(scores)
        del recent_scores[:-selection_smoothing]
        smoothed = tuple(float(np.mean(values)) for values in zip(*recent_scores))

        improved = best_key is None or smoothed > best_key
        row.update(validation_primary=scores[0], validation_secondary=scores[1],
                   validation_primary_smoothed=smoothed[0],
                   validation_secondary_smoothed=smoothed[1],
                   checkpoint_improved=improved)
        if improved:
            best_key = smoothed
            best_weights = model.embedding.weight.detach().clone()
            best_step = step
            stale_checks = 0
        else:
            stale_checks += 1
        return stale_checks >= patience

    validate(0, rows[0])
    model.train()
    for step in range(1, config.steps + 1):
        batch = batch_to_tensors(sampler.sample(batch_size=config.batch_size))
        optimizer.zero_grad(set_to_none=True)
        losses = model.bpr_objective(*batch, l2_weight=config.l2_weight)
        losses["loss"].backward()
        optimizer.step()
        row = {
            "step": step,
            "batch_bpr_loss_before_update": float(losses["bpr_loss"].detach().item()),
            "batch_total_loss_before_update": float(losses["loss"].detach().item()),
            "fixed_training_bpr_loss": fixed_batch_loss(),
        }
        rows.append(row)
        if step % validation_interval == 0 or step == config.steps:
            if validate(step, row):
                break
    if best_weights is not None:
        with torch.no_grad():
            model.embedding.weight.copy_(best_weights)
    model.eval()
    history = pd.DataFrame(rows)
    history.attrs.update(
        best_step=best_step if validation_callback is not None else int(rows[-1]["step"]),
        stopped_step=int(rows[-1]["step"]),
        early_stopped=int(rows[-1]["step"]) < config.steps,
        validation_selected=validation_callback is not None,
        selection_smoothing=int(selection_smoothing),
        patience=int(patience),
    )
    return model, history


def train_small_graph(
    graph: BipartiteGraphData,
    config: LightGCNTrainingConfig,
    **training_options,
) -> tuple[LightGCN, pd.DataFrame]:
    """Backward-compatible name for the educational small-graph trainer."""

    return train_lightgcn(graph, config, **training_options)
