#!/usr/bin/env python
"""Train and evaluate a local early-cycle baseline regressor."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from battery_aar.evaluation.metrics import regression_metrics
from battery_aar.features.early_cycle import build_early_cycle_features
from battery_aar.features.paper_features import build_paper_features, paper_feature_documentation
from battery_aar.models.baseline import (
    LOG_LIFE_MODEL_KINDS,
    SUPPORTED_MODEL_KINDS,
    predict,
    predict_log10_cycle_life,
    train_baseline,
)
from battery_aar.utils.io import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-cycle", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-kind", type=str, default="random_forest", choices=SUPPORTED_MODEL_KINDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = ensure_dir(args.out)
    metadata = pd.read_csv(args.processed_dir / "cell_metadata.csv")
    cycle_summary = pd.read_csv(args.processed_dir / "cycle_summary.csv")
    splits = pd.read_csv(args.processed_dir / "splits.csv")

    feature_family = "paper_features" if args.model_kind in LOG_LIFE_MODEL_KINDS else "generic_early_cycle"
    if feature_family == "paper_features":
        features = build_paper_features(metadata, cycle_summary, max_cycle=args.max_cycle)
    else:
        features = build_early_cycle_features(metadata, cycle_summary, max_cycle=args.max_cycle)
    features = features.merge(splits, on="cell_id", how="left")
    features["cycle_life"] = pd.to_numeric(features["cycle_life"], errors="coerce")
    labeled = features.loc[features["cycle_life"].notna()].copy()
    if labeled.empty:
        raise ValueError("no labeled cells available for supervised baseline training")
    if labeled["split"].isna().any():
        missing = labeled.loc[labeled["split"].isna(), "cell_id"].head().tolist()
        raise ValueError(f"some labeled cells are missing split assignments: {missing}")
    labeled = labeled[labeled["split"].isin(["train", "val", "test"])].copy()
    if labeled.empty:
        raise ValueError("no labeled train/val/test rows available after filtering")

    trained = train_baseline(labeled, model_kind=args.model_kind, seed=args.seed)
    labeled["y_pred"] = predict(trained, labeled)
    if trained.target_transform == "log10_cycle_life":
        labeled["log10_cycle_life"] = _log10_positive(labeled["cycle_life"])
        labeled["y_pred_log10"] = predict_log10_cycle_life(trained, labeled)

    metrics_by_split: dict[str, dict[str, float]] = {}
    for split, group in labeled.groupby("split"):
        metrics_by_split[str(split)] = regression_metrics(group["cycle_life"].to_numpy(), group["y_pred"].to_numpy())
    log_metrics_by_split = _log_metrics_by_split(labeled) if "y_pred_log10" in labeled else {}

    predictions = labeled[["cell_id", "split", "cycle_life", "y_pred"]].copy()
    predictions.to_csv(out / "predictions_all_splits.csv", index=False)
    predictions.loc[predictions["split"] == "val", ["cell_id", "y_pred"]].to_csv(
        out / "predictions_val.csv",
        index=False,
    )
    predictions.loc[predictions["split"] == "test", ["cell_id", "y_pred"]].to_csv(
        out / "predictions_test.csv",
        index=False,
    )
    predictions.loc[predictions["split"] == "test", ["cell_id", "y_pred"]].to_csv(
        out / "predictions.csv",
        index=False,
    )
    predictions.loc[predictions["split"] == "test", ["cell_id", "cycle_life"]].to_csv(
        out / "labels_test.csv",
        index=False,
    )
    labeled.to_csv(out / "features.csv", index=False)
    joblib.dump({"model": trained.model, "feature_columns": trained.feature_columns, "model_kind": trained.model_kind}, out / "model.joblib")
    write_json(out / "feature_columns.json", {"feature_columns": trained.feature_columns})
    write_json(out / "metrics.json", metrics_by_split)
    if log_metrics_by_split:
        write_json(out / "log_metrics.json", log_metrics_by_split)
    if feature_family == "paper_features":
        write_json(out / "paper_feature_documentation.json", paper_feature_documentation())
    write_json(
        out / "model_card.json",
        {
            "model_kind": trained.model_kind,
            "feature_family": feature_family,
            "target_transform": trained.target_transform,
            "max_cycle": args.max_cycle,
            "seed": args.seed,
            "n_labeled_cells": int(len(labeled)),
            "split_counts": labeled["split"].value_counts().astype(int).to_dict(),
            "feature_columns": trained.feature_columns,
            "intended_use": "local proof-of-concept baseline; not a validated scientific result",
            "data_warning": "demo data are synthetic unless processed-dir points to manually downloaded public data",
        },
    )
    print(f"Wrote baseline run to {out}")
    print(metrics_by_split)


def _log10_positive(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.Series(np.where(numeric > 0, np.log10(numeric), np.nan), index=values.index)


def _log_metrics_by_split(features: pd.DataFrame) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for split, group in features.groupby("split"):
        metrics[str(split)] = regression_metrics(
            group["log10_cycle_life"].to_numpy(),
            group["y_pred_log10"].to_numpy(),
        )
    return metrics


if __name__ == "__main__":
    main()
