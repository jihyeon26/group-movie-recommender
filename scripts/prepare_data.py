"""Run the reproducible MovieLens split, filtering, and pair pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from group_movie_recommender.config import PreprocessingConfig
from group_movie_recommender.pipeline import run_preprocessing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "movie_lens32m",
        help="Directory containing the extracted MovieLens 32M CSV files.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "preprocessing.json",
        help="JSON file containing preprocessing thresholds.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "processed_movielens32m",
        help="Directory for generated compressed CSV files and the manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PreprocessingConfig.from_json(args.config)
    manifest = run_preprocessing(args.data_dir, args.output_dir, config)
    print(json.dumps(manifest, indent=2))
    print(f"\nPreprocessing outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

