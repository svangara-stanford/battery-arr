#!/usr/bin/env python
"""Train and evaluate a local early-cycle baseline regressor."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from battery_aar.evaluation.metrics import regression_metrics
from battery_aar.features.early_cycle import build_early_cycle_features
from battery_aar.models.baseline import predict, train_baseline
from battery_aar.utils.io import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-cycle", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-kind", type=str, default="random_forest", choices=["dummy_mean", "ridge", "random_forest", "gradient_boosting"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = ensure_dir(args.out)
    metadata = pd.read_csv(args.processed_dir / "cell_metadata.csv")
    cycle_summary = pd.read_csv(args.processed_dir / "cycle_summary.csv")
    splits = pd.read_csv(args.processed_dir / "splits.csv")

    features = build_early_cycle_features(metadata, cycle_summary, max_cycle=args.max_cycle)
    features = features.merge(splits, on="cell_id", how="left")
    if features["split"].isna().any():
        raise ValueError("some cells are missing split assignments")

    trained = train_baseline(features, model_kind=args.model_kind, seed=args.seed)
    features = features.copy()
    features["y_pred"] = predict(trained, features)

    metrics_by_split: dict[str, dict[str, float]] = {}
    for split, group in features.groupby("split"):
        metrics_by_split[str(split)] = regression_metrics(group["cycle_life"].to_numpy(), group["y_pred"].to_numpy())

    predictions = features[["cell_id", "split", "cycle_life", "y_pred"]].copy()
    predictions.to_csv(out / "predictions_all_splits.csv", index=False)
    predictions.loc[predictions["split"] == "test", ["cell_id", "y_pred"]].to_csv(out / "predictions.csv", index=False)
    predictions.loc[predictions["split"] == "test", ["cell_id", "cycle_life"]].to_csv(out / "labels_test.csv", index=False)
    features.to_csv(out / "features.csv", index=False)
    joblib.dump({"model": trained.model, "feature_columns": trained.feature_columns, "model_kind": trained.model_kind}, out / "model.joblib")
    write_json(out / "metrics.json", metrics_by_split)
    write_json(
        out / "model_card.json",
        {
            "model_kind": trained.model_kind,
            "max_cycle": args.max_cycle,
            "seed": args.seed,
            "feature_columns": trained.feature_columns,
            "intended_use": "local proof-of-concept baseline; not a validated scientific result",
            "data_warning": "demo data are synthetic unless processed-dir points to manually downloaded public data",
        },
    )
    print(f"Wrote baseline run to {out}")
    print(metrics_by_split)


if __name__ == "__main__":
    main()
