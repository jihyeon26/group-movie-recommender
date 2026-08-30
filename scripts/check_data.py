"""Validate the local MovieLens 32M source files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from group_movie_recommender.preprocessing.data_check import validation_summary, validate_tables
from group_movie_recommender.shared.io import load_all_tables, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "movie_lens32m",
        help="Directory containing the extracted MovieLens 32M CSV files.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "data_validation_report.json",
        help="Destination for the JSON validation report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables = load_all_tables(args.data_dir)
    report = validate_tables(tables)
    write_json(report, args.report)
    print(validation_summary(report).to_string(index=False))
    print(f"\nValidation report: {args.report.resolve()}")
    if not report["passed_core_checks"]:
        raise SystemExit("Core data-quality checks failed")


if __name__ == "__main__":
    main()
