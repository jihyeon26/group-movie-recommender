"""Trainable PyTorch counterpart of the NumPy LightGCN reference."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .graph_data import BipartiteGraphData
from .lightgcn_math import normalized_edge_weights


class LightGCN(nn.Module):
    """Learn initial node embeddings through fixed, normalized graph propagation."""

    def __init__(
        self,
        graph: BipartiteGraphData,
        *,
        embedding_dim: int = 16,
        num_layers: int = 2,
        random_seed: int = 7,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or num_layers < 0:
            raise ValueError("embedding_dim must be positive and num_layers non-negative")

        self.num_users = graph.num_users
        self.num_movies = graph.num_movies
        self.num_layers = num_layers

        # Initialization is reproducible without changing the caller's random state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(random_seed)
            self.embedding = nn.Embedding(graph.num_nodes, embedding_dim)
            nn.init.normal_(self.embedding.weight, std=0.1)

        weights = normalized_edge_weights(graph.edge_index, num_nodes=graph.num_nodes)
        # Matrix row = target, column = source, matching the NumPy message direction.
        indices = torch.from_numpy(graph.edge_index[[1, 0]].copy()).long()
        adjacency = torch.sparse_coo_tensor(
            indices,
            torch.from_numpy(weights.astype(np.float32)),
            size=(graph.num_nodes, graph.num_nodes),
            check_invariants=True,
        ).coalesce()
        self.register_buffer("normalized_adjacency", adjacency)

    def propagate(self) -> tuple[Tensor, Tensor]:
        """Return final user and local-indexed movie embeddings."""

        current = self.embedding.weight
        total = current
        for _ in range(self.num_layers):
            current = torch.sparse.mm(self.normalized_adjacency, current)
            total = total + current
        final = total / (self.num_layers + 1)
        return final[:self.num_users], final[self.num_users:]

    def forward(self, user_indices: Tensor, movie_indices: Tensor) -> Tensor:
        """Score aligned user--movie pairs after graph propagation."""

        self._validate_indices(user_indices, self.num_users, "user_indices")
        self._validate_indices(movie_indices, self.num_movies, "movie_indices")
        if user_indices.shape != movie_indices.shape:
            raise ValueError("User and movie index arrays must have the same shape")
        users, movies = self.propagate()
        return (users[user_indices] * movies[movie_indices]).sum(dim=-1)

    @staticmethod
    def _validate_indices(indices: Tensor, size: int, name: str) -> None:
        if indices.ndim != 1 or indices.dtype != torch.long or indices.numel() == 0:
            raise ValueError(f"{name} must be a nonempty one-dimensional long tensor")
        if (indices < 0).any().item() or (indices >= size).any().item():
            raise ValueError(f"{name} contains an out-of-range index")

    def bpr_objective(
        self,
        user_indices: Tensor,
        positive_movie_indices: Tensor,
        negative_movie_indices: Tensor,
        *,
        l2_weight: float = 1e-4,
    ) -> dict[str, Tensor]:
        """Compute differentiable BPR loss with sampled initial-embedding L2."""

        if not np.isfinite(l2_weight) or l2_weight < 0:
            raise ValueError("l2_weight must be finite and non-negative")
        self._validate_indices(user_indices, self.num_users, "user_indices")
        self._validate_indices(positive_movie_indices, self.num_movies, "positive_movie_indices")
        self._validate_indices(negative_movie_indices, self.num_movies, "negative_movie_indices")
        if not user_indices.shape == positive_movie_indices.shape == negative_movie_indices.shape:
            raise ValueError("All BPR index arrays must have the same shape")

        # Propagate once per batch, then score both the positive and negative items.
        users, movies = self.propagate()
        user_vectors = users[user_indices]
        positive_scores = (user_vectors * movies[positive_movie_indices]).sum(dim=-1)
        negative_scores = (user_vectors * movies[negative_movie_indices]).sum(dim=-1)
        ranking_loss = F.softplus(negative_scores - positive_scores).mean()

        # Regularize layer-zero embeddings, not their propagated representations.
        user_initial = self.embedding(user_indices)
        positive_initial = self.embedding(self.num_users + positive_movie_indices)
        negative_initial = self.embedding(self.num_users + negative_movie_indices)
        l2_penalty = (
            user_initial.square().sum()
            + positive_initial.square().sum()
            + negative_initial.square().sum()
        ) / (2 * user_indices.numel())
        return {
            "loss": ranking_loss + l2_weight * l2_penalty,
            "bpr_loss": ranking_loss,
            "l2_penalty": l2_penalty,
        }
