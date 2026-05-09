"""Deterministic leakage-aware split utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitConfig:
    seed: int = 42
    val_fraction: float = 0.2
    test_fraction: float = 0.2
    group_column: str | None = None
    leave_one_group_value: str | None = None


def make_split_assignments(metadata: pd.DataFrame, config: SplitConfig | None = None) -> pd.DataFrame:
    """Create deterministic train/val/test assignments.

    If `group_column` is provided, all cells sharing that group are assigned to
    the same split. If `leave_one_group_value` is provided, that group becomes
    the test split and the remaining groups are split into train/val.
    """

    config = config or SplitConfig()
    if "cell_id" not in metadata.columns:
        raise ValueError("metadata must contain a 'cell_id' column")
    if not (0 <= config.val_fraction < 1 and 0 <= config.test_fraction < 1):
        raise ValueError("val_fraction and test_fraction must be in [0, 1)")
    if config.val_fraction + config.test_fraction >= 1 and config.leave_one_group_value is None:
        raise ValueError("val_fraction + test_fraction must be < 1")

    if config.leave_one_group_value is not None:
        if not config.group_column:
            raise ValueError("leave_one_group_value requires group_column")
        if config.group_column not in metadata.columns:
            raise ValueError(f"group_column {config.group_column!r} not in metadata")
        return _leave_one_group_split(metadata, config)

    rng = np.random.default_rng(config.seed)
    if config.group_column:
        if config.group_column not in metadata.columns:
            raise ValueError(f"group_column {config.group_column!r} not in metadata")
        units = metadata[[config.group_column]].drop_duplicates().reset_index(drop=True)
        unit_values = units[config.group_column].to_numpy()
        cell_units = metadata[["cell_id", config.group_column]].drop_duplicates()
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

    unit_col = config.group_column or "_unit"
    split = cell_units.copy()
    split["split"] = split[unit_col].map(assign)
    split = split[["cell_id", "split"]].drop_duplicates().reset_index(drop=True)
    _validate_no_overlap(split)
    return split


def _leave_one_group_split(metadata: pd.DataFrame, config: SplitConfig) -> pd.DataFrame:
    group_col = config.group_column
    assert group_col is not None
    test_mask = metadata[group_col].astype(str) == str(config.leave_one_group_value)
    if not test_mask.any():
        raise ValueError(f"leave-one group value {config.leave_one_group_value!r} not found")

    train_val = metadata.loc[~test_mask].copy()
    val_split = make_split_assignments(
        train_val,
        SplitConfig(
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
