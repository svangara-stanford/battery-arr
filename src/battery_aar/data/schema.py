"""Processed-data schema constants and validators."""

from __future__ import annotations

import numpy as np
import pandas as pd

CELL_METADATA_COLUMNS = [
    "cell_id",
    "batch_id",
    "cycle_life",
    "protocol_readable",
    "cc1",
    "cc2",
    "cc3",
    "cc4",
]

CYCLE_SUMMARY_COLUMNS = [
    "cell_id",
    "batch_id",
    "cycle_index",
    "discharge_capacity",
    "charge_capacity",
    "internal_resistance",
    "temperature_max",
    "temperature_mean",
    "temperature_min",
    "charge_time",
]

SPLIT_COLUMNS = ["cell_id", "split"]

REQUIRED_METADATA_COLUMNS = ["cell_id", "cycle_life"]
REQUIRED_SUMMARY_COLUMNS = ["cell_id", "cycle_index", "discharge_capacity"]


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return a copy with all requested columns present and ordered first."""

    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = np.nan
    extra = [c for c in out.columns if c not in columns]
    return out[columns + extra]


def validate_processed_tables(
    metadata: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    *,
    require_labels: bool = True,
) -> None:
    """Validate normalized metadata and cycle-summary tables.

    This intentionally checks only invariants that should hold for both the
    synthetic demo and the public MatR data. More dataset-specific QC belongs in
    preprocessing reports.
    """

    missing_meta = set(REQUIRED_METADATA_COLUMNS if require_labels else ["cell_id"]).difference(
        metadata.columns
    )
    missing_summary = set(REQUIRED_SUMMARY_COLUMNS).difference(cycle_summary.columns)
    if missing_meta:
        raise ValueError(f"metadata missing required columns: {sorted(missing_meta)}")
    if missing_summary:
        raise ValueError(f"cycle_summary missing required columns: {sorted(missing_summary)}")
    if metadata["cell_id"].isna().any():
        raise ValueError("metadata contains missing cell_id values")
    if metadata["cell_id"].duplicated().any():
        dupes = metadata.loc[metadata["cell_id"].duplicated(), "cell_id"].head().tolist()
        raise ValueError(f"metadata contains duplicate cell_id values, e.g. {dupes}")
    if require_labels and metadata["cycle_life"].isna().any():
        raise ValueError("metadata contains missing cycle_life labels")

    unknown_cells = set(cycle_summary["cell_id"].dropna()).difference(set(metadata["cell_id"]))
    if unknown_cells:
        sample = sorted(unknown_cells)[:5]
        raise ValueError(f"cycle_summary contains cells not present in metadata: {sample}")

    cycle_index = pd.to_numeric(cycle_summary["cycle_index"], errors="coerce")
    if cycle_index.isna().any():
        raise ValueError("cycle_summary contains missing cycle_index values")
    if (cycle_index < 1).any():
        raise ValueError("cycle_index must be one-indexed; normalize zero-indexed raw data first")
    grouped_min = cycle_summary.assign(_cycle_index=cycle_index).groupby("cell_id")["_cycle_index"].min()
    if not grouped_min.empty and (grouped_min != 1).all():
        raise ValueError("cycle_index does not start at 1 for any cell")
    if "discharge_capacity" in cycle_summary:
        q = pd.to_numeric(cycle_summary["discharge_capacity"], errors="coerce")
        if (q.dropna() <= 0).any():
            raise ValueError("discharge_capacity must be positive where present")


def validate_split_assignments(
    metadata: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    require_train_labels: bool = True,
) -> None:
    """Validate split rows and training-label availability.

    Splits are stored as one row per cell. The hidden-evaluator workflow relies
    on a strict partition: no cell may appear in more than one train/val/test
    split, and training rows must have labels in local supervised baselines.
    """

    missing = {"cell_id", "split"}.difference(splits.columns)
    if missing:
        raise ValueError(f"splits missing columns: {sorted(missing)}")
    if splits["cell_id"].duplicated().any():
        dupes = splits.loc[splits["cell_id"].duplicated(), "cell_id"].head().tolist()
        raise ValueError(f"splits contain duplicate cell assignments: {dupes}")
    unknown = set(splits["cell_id"].dropna()).difference(set(metadata["cell_id"]))
    if unknown:
        raise ValueError(f"splits contain cells not present in metadata: {sorted(unknown)[:5]}")
    valid_splits = {"train", "val", "test", "unlabeled"}
    invalid = set(splits["split"].dropna()).difference(valid_splits)
    if invalid:
        raise ValueError(f"splits contain invalid split names: {sorted(invalid)}")
    if require_train_labels:
        merged = splits.merge(metadata[["cell_id", "cycle_life"]], on="cell_id", how="left")
        train_labels = pd.to_numeric(
            merged.loc[merged["split"] == "train", "cycle_life"],
            errors="coerce",
        )
        if train_labels.isna().any():
            raise ValueError("training split contains rows without cycle_life labels")


def qc_summary(
    metadata: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    splits: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Build a compact JSON-serializable QC summary."""

    out: dict[str, object] = {
        "n_cells": int(metadata["cell_id"].nunique()) if "cell_id" in metadata else 0,
        "n_cycle_rows": int(len(cycle_summary)),
        "metadata_missing_by_column": metadata.isna().sum().astype(int).to_dict(),
        "cycle_summary_missing_by_column": cycle_summary.isna().sum().astype(int).to_dict(),
    }
    if "batch_id" in metadata:
        out["cells_by_batch"] = metadata["batch_id"].fillna("UNKNOWN").value_counts().astype(int).to_dict()
    if "batch_id" in cycle_summary:
        out["cycle_rows_by_batch"] = (
            cycle_summary["batch_id"].fillna("UNKNOWN").value_counts().astype(int).to_dict()
        )
    if "cycle_life" in metadata:
        labels = pd.to_numeric(metadata["cycle_life"], errors="coerce").dropna()
        out["cycle_life"] = {
            "count": int(labels.size),
            "mean": float(labels.mean()) if labels.size else None,
            "std": float(labels.std(ddof=0)) if labels.size else None,
            "min": float(labels.min()) if labels.size else None,
            "max": float(labels.max()) if labels.size else None,
        }
    if {"cell_id", "cycle_index"}.issubset(cycle_summary.columns):
        counts = cycle_summary.groupby("cell_id")["cycle_index"].nunique()
        out["cycles_per_cell"] = {
            "min": int(counts.min()) if len(counts) else 0,
            "median": float(counts.median()) if len(counts) else 0.0,
            "max": int(counts.max()) if len(counts) else 0,
        }
        out["cycles_per_cell_by_id"] = counts.astype(int).to_dict()
    if splits is not None and "split" in splits:
        out["split_counts"] = splits["split"].fillna("UNKNOWN").value_counts().astype(int).to_dict()
    return out
