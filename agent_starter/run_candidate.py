#!/usr/bin/env python
"""Run a minimal non-LLM candidate configuration through the evaluator path."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml

from battery_aar.evaluation.evaluator import evaluate_predictions
from battery_aar.features.early_cycle import build_early_cycle_features
from battery_aar.models.baseline import predict, select_feature_columns, train_baseline
from battery_aar.utils.io import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_candidate(args.processed_dir, args.config, args.out)
    print(result)


def run_candidate(processed_dir: Path, config_path: Path, output_dir: Path) -> dict[str, Any]:
    """Train and locally evaluate one candidate YAML config."""

    output_dir = ensure_dir(output_dir)
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"candidate config must be a mapping: {config_path}")

    max_cycle = int(config.get("max_cycle", 100))
    seed = int(config.get("seed", 42))
    model_kind = str(config["model_kind"])

    metadata = pd.read_csv(processed_dir / "cell_metadata.csv")
    cycle_summary = pd.read_csv(processed_dir / "cycle_summary.csv")
    splits = pd.read_csv(processed_dir / "splits.csv")
    features = build_early_cycle_features(metadata, cycle_summary, max_cycle=max_cycle)
    features = features.merge(splits, on="cell_id", how="left")
    features["cycle_life"] = pd.to_numeric(features["cycle_life"], errors="coerce")
    labeled = features.loc[
        features["cycle_life"].notna() & features["split"].isin(["train", "val", "test"])
    ].copy()
    if labeled.empty:
        raise ValueError("candidate has no labeled train/val/test rows")
    feature_columns = _candidate_feature_columns(labeled, config)
    trained = train_baseline(
        labeled,
        model_kind=model_kind,
        seed=seed,
        feature_columns=feature_columns,
    )
    labeled["y_pred"] = predict(trained, labeled)

    predictions = labeled[["cell_id", "split", "cycle_life", "y_pred"]].copy()
    predictions.to_csv(output_dir / "predictions_all_splits.csv", index=False)
    submission = predictions.loc[predictions["split"] == "test", ["cell_id", "y_pred"]].copy()
    if submission.empty:
        raise ValueError("candidate evaluation requires at least one test prediction")
    truth = predictions.loc[predictions["split"] == "test", ["cell_id", "cycle_life"]].copy()
    submission.to_csv(output_dir / "predictions.csv", index=False)
    truth.to_csv(output_dir / "labels_test.csv", index=False)

    metrics = evaluate_predictions(truth, submission)
    joblib.dump(
        {
            "model": trained.model,
            "feature_columns": trained.feature_columns,
            "model_kind": trained.model_kind,
        },
        output_dir / "model.joblib",
    )
    write_json(output_dir / "metrics.json", {"test": metrics})
    write_json(output_dir / "feature_columns.json", {"feature_columns": trained.feature_columns})
    write_json(
        output_dir / "model_card.json",
        {
            "model_kind": trained.model_kind,
            "max_cycle": max_cycle,
            "seed": seed,
            "feature_columns": trained.feature_columns,
            "intended_use": "agent starter candidate; local non-LLM smoke test",
        },
    )
    result = {
        "candidate_name": str(config.get("name", config_path.stem)),
        "model_kind": trained.model_kind,
        "max_cycle": max_cycle,
        "seed": seed,
        "n_features": len(trained.feature_columns),
        "n_test": int(len(submission)),
        "metrics": metrics,
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "candidate_result.json", result)
    return result


def _candidate_feature_columns(features: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    available = select_feature_columns(features)
    prefixes = [str(value) for value in config.get("feature_include_prefixes", [])]
    explicit = {str(value) for value in config.get("feature_include_columns", [])}
    if not prefixes and not explicit:
        return available
    selected = [
        col
        for col in available
        if col in explicit or any(col.startswith(prefix) for prefix in prefixes)
    ]
    if not selected:
        raise ValueError("candidate feature filters selected no numeric columns")
    return selected


if __name__ == "__main__":
    main()
