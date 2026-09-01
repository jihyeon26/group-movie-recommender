"""Prepare, train, and evaluate the frozen user-disjoint final report experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from group_movie_recommender.pipelines.final_holdout_experiment import (
    evaluate_final_holdout_once,
    prepare_final_holdout,
    train_final_holdout_models,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "train", "test"))
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT_ROOT / "dataset/movie_lens32m"
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/processed_movielens32m",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/final_holdout_experiment",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/final_holdout_experiment.json",
    )
    parser.add_argument(
        "--preprocessing",
        type=Path,
        default=PROJECT_ROOT / "configs/preprocessing.json",
    )
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare_final_holdout(
            data_dir=args.data_dir,
            processed_dir=args.processed_dir,
            output_root=args.output_root,
            config_path=args.config,
            preprocessing_path=args.preprocessing,
        )
    elif args.phase == "train":
        train_final_holdout_models(data_dir=args.data_dir, output_root=args.output_root)
    else:
        evaluate_final_holdout_once(data_dir=args.data_dir, output_root=args.output_root)


if __name__ == "__main__":
    main()
