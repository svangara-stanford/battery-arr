#!/usr/bin/env python
"""Run a small real-data baseline matrix for reviewer-facing evidence."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from battery_aar.data.matr_io import load_raw_batches
from battery_aar.data.schema import (
    qc_summary,
    validate_processed_tables,
    validate_split_assignments,
)
from battery_aar.data.split import SplitConfig, make_split_assignments
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

DEFAULT_SPLIT_MODES = ("random", "batch", "protocol", "leave_one_batch_out")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/chueh_toyota_fast_charge"))
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed/chueh_toyota_fast_charge"),
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/chueh_toyota_phase1"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--max-cycle", type=int, default=100)
    parser.add_argument("--first-n-cycles", type=int, default=100)
    parser.add_argument(
        "--max-cells-per-batch",
        type=int,
        default=4,
        help="Reviewer-smoke default. Use 0 to process every cell in each batch.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--leave-one-batch-id", type=str, default=None)
    parser.add_argument("--model-kinds", nargs="+", default=list(SUPPORTED_MODEL_KINDS))
    parser.add_argument("--split-modes", nargs="+", default=list(DEFAULT_SPLIT_MODES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_cells = args.max_cells_per_batch if args.max_cells_per_batch > 0 else None
    rows = run_matrix(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        runs_dir=args.runs_dir,
        reports_dir=args.reports_dir,
        model_kinds=args.model_kinds,
        split_modes=args.split_modes,
        max_cycle=args.max_cycle,
        first_n_cycles=args.first_n_cycles,
        max_cells_per_batch=max_cells,
        seed=args.seed,
        leave_one_batch_id=args.leave_one_batch_id,
    )
    print(_markdown_table(pd.DataFrame(rows)))


def run_matrix(
    *,
    raw_dir: Path,
    processed_dir: Path,
    runs_dir: Path,
    reports_dir: Path,
    model_kinds: list[str],
    split_modes: list[str],
    max_cycle: int,
    first_n_cycles: int,
    max_cells_per_batch: int | None,
    seed: int,
    leave_one_batch_id: str | None = None,
) -> list[dict[str, Any]]:
    """Preprocess raw data once, then run the requested split/model matrix."""

    processed_dir = ensure_dir(processed_dir)
    runs_dir = ensure_dir(runs_dir)
    reports_dir = ensure_dir(reports_dir)

    loaded = load_raw_batches(
        raw_dir,
        max_cells_per_batch=max_cells_per_batch,
        first_n_cycles=first_n_cycles,
    )
    validate_processed_tables(loaded.metadata, loaded.cycle_summary, require_labels=False)
    loaded.metadata.to_csv(processed_dir / "cell_metadata.csv", index=False)
    loaded.cycle_summary.to_csv(processed_dir / "cycle_summary.csv", index=False)
    write_json(processed_dir / "qc_summary.json", qc_summary(loaded.metadata, loaded.cycle_summary))

    rows: list[dict[str, Any]] = []
    for split_mode in split_modes:
        split_result = _make_split_for_matrix(
            loaded.metadata,
            split_mode=split_mode,
            seed=seed,
            leave_one_batch_id=leave_one_batch_id,
        )
        if isinstance(split_result, str):
            for model_kind in model_kinds:
                rows.append(
                    _summary_row(
                        split_mode=split_mode,
                        model_kind=model_kind,
                        status="failed",
                        output_dir=runs_dir / f"{split_mode}_{model_kind}",
                        failure_reason=split_result,
                    )
                )
            continue

        splits = split_result
        split_path = processed_dir / f"splits_{split_mode}.csv"
        splits.to_csv(split_path, index=False)
        write_json(
            processed_dir / f"qc_summary_{split_mode}.json",
            qc_summary(loaded.metadata, loaded.cycle_summary, splits),
        )
        counts = _split_counts_for_labeled_cells(loaded.metadata, splits)
        for model_kind in model_kinds:
            output_dir = runs_dir / f"{split_mode}_{model_kind}"
            try:
                metrics = _run_baseline(
                    metadata=loaded.metadata,
                    cycle_summary=loaded.cycle_summary,
                    splits=splits,
                    output_dir=output_dir,
                    model_kind=model_kind,
                    max_cycle=max_cycle,
                    seed=seed,
                )
                test_metrics = metrics.get("test", {})
                rows.append(
                    _summary_row(
                        split_mode=split_mode,
                        model_kind=model_kind,
                        status="ok",
                        output_dir=output_dir,
                        counts=counts,
                        metrics=test_metrics,
                    )
                )
            except Exception as exc:
                rows.append(
                    _summary_row(
                        split_mode=split_mode,
                        model_kind=model_kind,
                        status="failed",
                        output_dir=output_dir,
                        counts=counts,
                        failure_reason=str(exc),
                    )
                )

    summary = pd.DataFrame(rows)
    summary.to_csv(reports_dir / "real_data_baseline_matrix.csv", index=False)
    (reports_dir / "real_data_baseline_matrix.md").write_text(
        "# Real Data Baseline Matrix\n\n"
        "This table is generated by `scripts/run_real_data_baseline_matrix.py`. "
        "Metrics are test-split metrics where a test split exists.\n\n"
        + _markdown_table(summary)
        + "\n",
    )
    return rows


def _make_split_for_matrix(
    metadata: pd.DataFrame,
    *,
    split_mode: str,
    seed: int,
    leave_one_batch_id: str | None,
) -> pd.DataFrame | str:
    labels = pd.to_numeric(metadata["cycle_life"], errors="coerce")
    labeled = metadata.loc[labels.notna()].drop_duplicates("cell_id").copy()
    if labeled.empty:
        return "no labeled cells available for supervised matrix run"
    try:
        _validate_split_units(labeled, split_mode)
        value = leave_one_batch_id
        if split_mode == "leave_one_batch_out":
            value = value or _default_holdout_batch(labeled)
        splits = make_split_assignments(
            labeled,
            SplitConfig(
                mode=split_mode,
                seed=seed,
                leave_one_group_value=value,
            ),
        )
        unlabeled = metadata.loc[labels.isna(), ["cell_id"]].drop_duplicates().copy()
        if not unlabeled.empty:
            unlabeled["split"] = "unlabeled"
            splits = pd.concat([splits, unlabeled], ignore_index=True)
        validate_split_assignments(metadata, splits, require_train_labels=True)
        return splits
    except Exception as exc:
        return str(exc)


def _validate_split_units(labeled: pd.DataFrame, split_mode: str) -> None:
    if split_mode == "random":
        n_units = labeled["cell_id"].nunique()
        if n_units < 3:
            raise ValueError(f"random split requires at least 3 labeled cells; found {n_units}")
    elif split_mode == "batch":
        n_units = labeled["batch_id"].nunique()
        if n_units < 3:
            raise ValueError(f"batch split requires at least 3 labeled batches; found {n_units}")
    elif split_mode == "protocol":
        n_units = labeled["protocol_readable"].nunique()
        if n_units < 3:
            raise ValueError(f"protocol split requires at least 3 labeled protocols; found {n_units}")
    elif split_mode == "leave_one_batch_out":
        n_units = labeled["batch_id"].nunique()
        if n_units < 2:
            raise ValueError(
                f"leave_one_batch_out requires at least 2 labeled batches; found {n_units}"
            )
    else:
        raise ValueError(f"unknown split mode {split_mode!r}")


def _default_holdout_batch(labeled: pd.DataFrame) -> str:
    counts = labeled["batch_id"].astype(str).value_counts()
    return str(counts.sort_index().index[-1])


def _run_baseline(
    *,
    metadata: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    splits: pd.DataFrame,
    output_dir: Path,
    model_kind: str,
    max_cycle: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    output_dir = ensure_dir(output_dir)
    feature_family = "paper_features" if model_kind in LOG_LIFE_MODEL_KINDS else "generic_early_cycle"
    if feature_family == "paper_features":
        features = build_paper_features(metadata, cycle_summary, max_cycle=max_cycle)
    else:
        features = build_early_cycle_features(metadata, cycle_summary, max_cycle=max_cycle)
    features = features.merge(splits, on="cell_id", how="left")
    features["cycle_life"] = pd.to_numeric(features["cycle_life"], errors="coerce")
    labeled = features.loc[
        features["cycle_life"].notna() & features["split"].isin(["train", "val", "test"])
    ].copy()
    if labeled.empty:
        raise ValueError("no labeled train/val/test rows after feature construction")

    trained = train_baseline(labeled, model_kind=model_kind, seed=seed)
    labeled["y_pred"] = predict(trained, labeled)
    if trained.target_transform == "log10_cycle_life":
        labeled["log10_cycle_life"] = _log10_positive(labeled["cycle_life"])
        labeled["y_pred_log10"] = predict_log10_cycle_life(trained, labeled)
    metrics_by_split: dict[str, dict[str, float]] = {}
    for split, group in labeled.groupby("split"):
        metrics_by_split[str(split)] = regression_metrics(
            group["cycle_life"].to_numpy(),
            group["y_pred"].to_numpy(),
        )
    log_metrics_by_split = _log_metrics_by_split(labeled) if "y_pred_log10" in labeled else {}

    predictions = labeled[["cell_id", "split", "cycle_life", "y_pred"]].copy()
    predictions.to_csv(output_dir / "predictions_all_splits.csv", index=False)
    for split in ("val", "test"):
        predictions.loc[predictions["split"] == split, ["cell_id", "y_pred"]].to_csv(
            output_dir / f"predictions_{split}.csv",
            index=False,
        )
    predictions.loc[predictions["split"] == "test", ["cell_id", "y_pred"]].to_csv(
        output_dir / "predictions.csv",
        index=False,
    )
    predictions.loc[predictions["split"] == "test", ["cell_id", "cycle_life"]].to_csv(
        output_dir / "labels_test.csv",
        index=False,
    )
    labeled.to_csv(output_dir / "features.csv", index=False)
    joblib.dump(
        {
            "model": trained.model,
            "feature_columns": trained.feature_columns,
            "model_kind": trained.model_kind,
        },
        output_dir / "model.joblib",
    )
    write_json(output_dir / "feature_columns.json", {"feature_columns": trained.feature_columns})
    write_json(output_dir / "metrics.json", metrics_by_split)
    if log_metrics_by_split:
        write_json(output_dir / "log_metrics.json", log_metrics_by_split)
    if feature_family == "paper_features":
        write_json(output_dir / "paper_feature_documentation.json", paper_feature_documentation())
    write_json(
        output_dir / "model_card.json",
        {
            "model_kind": trained.model_kind,
            "feature_family": feature_family,
            "target_transform": trained.target_transform,
            "max_cycle": max_cycle,
            "seed": seed,
            "n_labeled_cells": int(len(labeled)),
            "split_counts": labeled["split"].value_counts().astype(int).to_dict(),
            "intended_use": "phase-1 real-data baseline matrix; not a scientific claim",
            "data_warning": "raw MatR data are manually downloaded and not committed",
        },
    )
    return metrics_by_split


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


def _split_counts_for_labeled_cells(metadata: pd.DataFrame, splits: pd.DataFrame) -> dict[str, int]:
    labeled = metadata.loc[pd.to_numeric(metadata["cycle_life"], errors="coerce").notna(), ["cell_id"]]
    merged = labeled.merge(splits, on="cell_id", how="left")
    counts = merged["split"].value_counts().astype(int).to_dict()
    return {name: int(counts.get(name, 0)) for name in ("train", "val", "test")}


def _summary_row(
    *,
    split_mode: str,
    model_kind: str,
    status: str,
    output_dir: Path,
    counts: dict[str, int] | None = None,
    metrics: dict[str, float] | None = None,
    failure_reason: str = "",
) -> dict[str, Any]:
    counts = counts or {}
    metrics = metrics or {}
    return {
        "split_mode": split_mode,
        "model_kind": model_kind,
        "status": status,
        "n_train": counts.get("train", np.nan),
        "n_val": counts.get("val", np.nan),
        "n_test": counts.get("test", np.nan),
        "rmse": metrics.get("rmse", np.nan),
        "mae": metrics.get("mae", np.nan),
        "r2": metrics.get("r2", np.nan),
        "output_dir": str(output_dir),
        "failure_reason": failure_reason,
    }


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    display = df.copy()
    for col in ("rmse", "mae", "r2"):
        if col in display:
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
    display = display.fillna("")
    headers = [str(c) for c in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        values = [str(row[c]).replace("\n", " ") for c in display.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
