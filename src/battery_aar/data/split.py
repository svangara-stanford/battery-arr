"""Deterministic leakage-aware split utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitConfig:
    mode: str = "random"
    seed: int = 42
    val_fraction: float = 0.2
    test_fraction: float = 0.2
    group_column: str | None = None
    leave_one_group_value: str | None = None


def make_split_assignments(metadata: pd.DataFrame, config: SplitConfig | None = None) -> pd.DataFrame:
    """Create deterministic train/val/test assignments.

    Supported modes:
    - `random`: random cell split for smoke tests.
    - `batch`: group split by `batch_id`.
    - `protocol`: group split by `protocol_readable`.
    - `protocol_cluster`: group split by exact C-rate tuple when available.
    - `leave_one_batch_out`: hold one batch out as the test split.

    Final scientific claims should use group/transfer splits, not only the
    random smoke-test split.
    """

    config = config or SplitConfig()
    if "cell_id" not in metadata.columns:
        raise ValueError("metadata must contain a 'cell_id' column")
    if not (0 <= config.val_fraction < 1 and 0 <= config.test_fraction < 1):
        raise ValueError("val_fraction and test_fraction must be in [0, 1)")
    if config.val_fraction + config.test_fraction >= 1 and config.leave_one_group_value is None:
        raise ValueError("val_fraction + test_fraction must be < 1")

    metadata = metadata.drop_duplicates("cell_id").reset_index(drop=True).copy()
    group_column = _resolve_group_column(metadata, config)
    if config.mode == "leave_one_batch_out" and config.leave_one_group_value is None:
        raise ValueError("leave_one_batch_out requires leave_one_group_value")

    if config.leave_one_group_value is not None:
        if not group_column:
            raise ValueError("leave_one_group_value requires a group split mode or group_column")
        return _leave_one_group_split(metadata, config, group_column)

    rng = np.random.default_rng(config.seed)
    if group_column:
        units = metadata[[group_column]].drop_duplicates().reset_index(drop=True)
        unit_values = units[group_column].to_numpy()
        cell_units = metadata[["cell_id", group_column]].drop_duplicates()
    else:
        unit_values = metadata["cell_id"].drop_duplicates().to_numpy()
        cell_units = pd.DataFrame({"cell_id": unit_values, "_unit": unit_values})

    shuffled = unit_values.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = _split_count(n, config.test_fraction, prefer_nonzero=n >= 3)
    n_val = _split_count(n - n_test, config.val_fraction, prefer_nonzero=(n - n_test) >= 3)

    test_units = set(shuffled[:n_test])
    val_units = set(shuffled[n_test : n_test + n_val])

    def assign(unit: object) -> str:
        if unit in test_units:
            return "test"
        if unit in val_units:
            return "val"
        return "train"

    unit_col = group_column or "_unit"
    split = cell_units.copy()
    split["split"] = split[unit_col].map(assign)
    split = split[["cell_id", "split"]].drop_duplicates().reset_index(drop=True)
    _validate_no_overlap(split)
    return split


def _resolve_group_column(metadata: pd.DataFrame, config: SplitConfig) -> str | None:
    mode = config.mode.lower()
    if config.group_column:
        if config.group_column not in metadata.columns:
            raise ValueError(f"group_column {config.group_column!r} not in metadata")
        return config.group_column
    if mode == "random":
        return None
    if mode in {"batch", "leave_one_batch_out"}:
        if "batch_id" not in metadata.columns:
            raise ValueError("batch split mode requires metadata.batch_id")
        return "batch_id"
    if mode == "protocol":
        if "protocol_readable" not in metadata.columns:
            raise ValueError("protocol split mode requires metadata.protocol_readable")
        return "protocol_readable"
    if mode == "protocol_cluster":
        return _add_protocol_cluster(metadata)
    raise ValueError(f"unknown split mode {config.mode!r}")


def _add_protocol_cluster(metadata: pd.DataFrame) -> str:
    cluster_col = "_protocol_cluster"
    if {"cc1", "cc2", "cc3", "cc4"}.issubset(metadata.columns):
        rates = metadata[["cc1", "cc2", "cc3", "cc4"]].apply(pd.to_numeric, errors="coerce")
        metadata[cluster_col] = rates.round(1).astype(str).agg("_".join, axis=1)
    elif "protocol_readable" in metadata.columns:
        metadata[cluster_col] = metadata["protocol_readable"].astype(str)
    else:
        raise ValueError("protocol_cluster split mode requires C-rate columns or protocol_readable")
    return cluster_col


def _leave_one_group_split(
    metadata: pd.DataFrame,
    config: SplitConfig,
    group_col: str,
) -> pd.DataFrame:
    test_mask = metadata[group_col].astype(str) == str(config.leave_one_group_value)
    if not test_mask.any():
        raise ValueError(f"leave-one group value {config.leave_one_group_value!r} not found")

    train_val = metadata.loc[~test_mask].copy()
    val_split = make_split_assignments(
        train_val,
        SplitConfig(
            mode="random" if group_col == "_protocol_cluster" else "batch",
            seed=config.seed,
            val_fraction=config.val_fraction,
            test_fraction=0.0,
            group_column=group_col,
        ),
    )
    val_split.loc[val_split["split"] == "test", "split"] = "val"
    test_split = metadata.loc[test_mask, ["cell_id"]].copy()
    test_split["split"] = "test"
    split = pd.concat([val_split, test_split], ignore_index=True).drop_duplicates("cell_id")
    _validate_no_overlap(split)
    return split.reset_index(drop=True)


def _split_count(n: int, fraction: float, *, prefer_nonzero: bool) -> int:
    if n <= 1 or fraction <= 0:
        return 0
    count = int(round(n * fraction))
    if prefer_nonzero:
        count = max(1, count)
    return min(count, max(n - 1, 0))


def _validate_no_overlap(split: pd.DataFrame) -> None:
    if split["cell_id"].duplicated().any():
        dupes = split.loc[split["cell_id"].duplicated(), "cell_id"].head().tolist()
        raise ValueError(f"split contains duplicate cell assignments: {dupes}")
