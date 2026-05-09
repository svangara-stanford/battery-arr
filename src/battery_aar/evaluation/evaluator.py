"""Submission evaluator helpers."""

from __future__ import annotations

import pandas as pd

from battery_aar.evaluation.metrics import regression_metrics


def evaluate_predictions(truth: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, float]:
    """Evaluate predictions with columns `cell_id` and `y_pred` against hidden truth."""

    required_truth = {"cell_id", "cycle_life"}
    required_pred = {"cell_id", "y_pred"}
    if missing := required_truth.difference(truth.columns):
        raise ValueError(f"truth missing columns: {sorted(missing)}")
    if missing := required_pred.difference(predictions.columns):
        raise ValueError(f"predictions missing columns: {sorted(missing)}")
    extra_pred = set(predictions.columns).difference(required_pred)
    if extra_pred:
        raise ValueError(
            "predictions must contain only cell_id and y_pred; "
            f"unexpected columns: {sorted(extra_pred)}"
        )
    if predictions["cell_id"].duplicated().any():
        raise ValueError("predictions contain duplicate cell_id values")
    merged = truth[["cell_id", "cycle_life"]].merge(predictions[["cell_id", "y_pred"]], on="cell_id", how="left")
    if merged["y_pred"].isna().any():
        missing_cells = merged.loc[merged["y_pred"].isna(), "cell_id"].head().tolist()
        raise ValueError(f"missing predictions for cells: {missing_cells}")
    return regression_metrics(merged["cycle_life"].to_numpy(), merged["y_pred"].to_numpy())
