"""Inspectable mini-batch training on small graphs, before MovieLens scaling."""

from __future__ import annotations

from dataclasses import dataclass

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


def train_small_graph(
    graph: BipartiteGraphData,
    config: LightGCNTrainingConfig,
) -> tuple[LightGCN, pd.DataFrame]:
    """Train on sampled positives; monitor a fixed training diagnostic batch.

    The monitor is NOT a validation/test set or a recommendation metric. This CPU
    reference propagates over the complete graph each update and is not yet a
    throughput-tuned MovieLens 32M trainer.
    """

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
    model.train()
    for step in range(1, config.steps + 1):
        batch = batch_to_tensors(sampler.sample(batch_size=config.batch_size))
        optimizer.zero_grad(set_to_none=True)
        losses = model.bpr_objective(*batch, l2_weight=config.l2_weight)
        losses["loss"].backward()
        optimizer.step()
        rows.append({
            "step": step,
            "batch_bpr_loss_before_update": float(losses["bpr_loss"].detach().item()),
            "batch_total_loss_before_update": float(losses["loss"].detach().item()),
            "fixed_training_bpr_loss": fixed_batch_loss(),
        })
    model.eval()
    return model, pd.DataFrame(rows)
