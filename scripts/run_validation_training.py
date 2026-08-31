"""Run selection first, then separately evaluate the frozen models on test."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from group_movie_recommender.pipelines.validation_experiment import run_frozen_test, run_selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("select", "test"))
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "dataset/movie_lens32m")
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "outputs/processed_movielens32m")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/validation_training.json")
    parser.add_argument("--preprocessing-config", type=Path, default=PROJECT_ROOT / "configs/preprocessing.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/validation_selected_model")
    args = parser.parse_args()
    if args.phase == "select":
        run_selection(data_dir=args.data_dir, processed_dir=args.processed_dir,
                      output_root=args.output_root, config_path=args.config,
                      preprocessing_path=args.preprocessing_config)
    else:
        run_frozen_test(data_dir=args.data_dir, output_root=args.output_root)


if __name__ == "__main__":
    main()
