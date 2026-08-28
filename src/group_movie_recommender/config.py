"""Configuration objects for the MovieLens preprocessing pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PreprocessingConfig:
    """Reproducible thresholds used by the preprocessing pipeline."""

    positive_threshold: float = 4.0
    train_end: str = "2019-01-01"
    validation_end: str = "2020-01-01"
    training_min_positives: int = 5
    evaluation_min_train_ratings: int = 100
    evaluation_min_train_positives: int = 20
    evaluation_min_validation_positives: int = 3
    evaluation_min_test_positives: int = 5
    warm_item_min_train_positives: int = 5
    pair_min_co_rated: int = 20
    pair_similarity_quantile: float = 0.25
    pair_candidate_sample_size: int = 100_000
    max_pairs_per_user: int = 3
    genre_prior_strength: float = 10.0
    random_seed: int = 20_260_828

    @classmethod
    def from_json(cls, path: str | Path) -> "PreprocessingConfig":
        """Load configuration values from a JSON file."""

        with Path(path).open("r", encoding="utf-8") as handle:
            values: dict[str, Any] = json.load(handle)
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)

    @staticmethod
    def _utc_timestamp(date_text: str) -> int:
        date_value = datetime.strptime(date_text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return int(date_value.timestamp())

    @property
    def train_end_timestamp(self) -> int:
        """Return the exclusive UTC training cutoff as epoch seconds."""

        return self._utc_timestamp(self.train_end)

    @property
    def validation_end_timestamp(self) -> int:
        """Return the exclusive UTC validation cutoff as epoch seconds."""

        return self._utc_timestamp(self.validation_end)

