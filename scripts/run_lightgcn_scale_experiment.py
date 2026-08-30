"""Run controlled LightGCN training and validation stages reproducibly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from group_movie_recommender.evaluation.experiment_summary import build_stage_rows
from group_movie_recommender.shared.io import write_csv_gzip, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/lightgcn_scale_experiment.json",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["smoke"],
        help="Named stages to run in the order listed in the experiment config.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "dataset/movie_lens32m",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/processed_movielens32m",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/lightgcn_scale_experiment",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Rebuild the comparison from completed stages without retraining.",
    )
    args = parser.parse_args()

    experiment = json.loads(args.config.read_text(encoding="utf-8"))
    configured_stages = experiment.get("stages", {})
    missing = [stage for stage in args.stages if stage not in configured_stages]
    if missing:
        raise ValueError(
            f"Unknown stages {missing}; available stages: {list(configured_stages)}"
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        for stage in args.stages:
            run_stage(
                stage,
                configured_stages[stage],
                data_dir=args.data_dir,
                processed_dir=args.processed_dir,
                output_root=args.output_root,
            )
    comparison = collect_completed_stages(args.output_root, list(configured_stages))
    write_csv_gzip(comparison, args.output_root / "scale_comparison.csv.gz")
    print("\nCompleted-stage comparison:")
    print(comparison.to_string(index=False))


def run_stage(
    stage: str,
    settings: dict[str, object],
    *,
    data_dir: Path,
    processed_dir: Path,
    output_root: Path,
) -> None:
    """Train and validate one named scale while recording elapsed time."""

    stage_dir = output_root / stage
    model_dir = stage_dir / "model"
    validation_dir = stage_dir / "validation"
    resolved_config = stage_dir / "resolved_training_config.json"
    stage_dir.mkdir(parents=True, exist_ok=True)
    write_json(settings, resolved_config)

    train_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/train_lightgcn_subset.py"),
        "--processed-dir",
        str(processed_dir),
        "--config",
        str(resolved_config),
        "--output-dir",
        str(model_dir),
    ]
    validation_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/evaluate_lightgcn_subset.py"),
        "--data-dir",
        str(data_dir),
        "--embedding-dir",
        str(model_dir),
        "--output-dir",
        str(validation_dir),
    ]

    print(f"\n=== Stage: {stage} ===", flush=True)
    start = time.perf_counter()
    subprocess.run(train_command, check=True)
    trained_at = time.perf_counter()
    subprocess.run(validation_command, check=True)
    finished_at = time.perf_counter()
    write_json(
        {
            "stage": stage,
            "training_seconds": trained_at - start,
            "validation_seconds": finished_at - trained_at,
            "total_seconds": finished_at - start,
        },
        stage_dir / "runtime.json",
    )


def collect_completed_stages(output_root: Path, stages: list[str]) -> pd.DataFrame:
    """Collect only stages with both training and validation reports."""

    rows: list[dict[str, object]] = []
    for stage in stages:
        training_path = output_root / stage / "model/summary.json"
        validation_path = output_root / stage / "validation/validation_report.json"
        if not training_path.exists() or not validation_path.exists():
            continue
        training_summary = json.loads(training_path.read_text(encoding="utf-8"))
        validation_report = json.loads(validation_path.read_text(encoding="utf-8"))
        rows.extend(build_stage_rows(stage, training_summary, validation_report))
    if not rows:
        raise ValueError("No completed experiment stages were found")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
