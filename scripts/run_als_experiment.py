"""Select implicit ALS on validation and optionally run an exploratory frozen test."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from group_movie_recommender.pipelines.als_experiment import (
    run_als_exploratory_test, run_als_selection,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("select", "test"))
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "dataset/movie_lens32m")
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "outputs/processed_movielens32m")
    parser.add_argument("--base-root", type=Path, default=PROJECT_ROOT / "outputs/validation_selected_model")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/als_experiment")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/als_experiment.json")
    args = parser.parse_args()
    if args.phase == "select":
        run_als_selection(data_dir=args.data_dir, processed_dir=args.processed_dir,
                          base_root=args.base_root, output_root=args.output_root,
                          config_path=args.config)
    else:
        run_als_exploratory_test(data_dir=args.data_dir, processed_dir=args.processed_dir,
                                 base_root=args.base_root, output_root=args.output_root)


if __name__ == "__main__":
    main()
