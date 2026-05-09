from __future__ import annotations

import pandas as pd
import pytest

from battery_aar.data.schema import validate_processed_tables, validate_split_assignments


def test_validate_processed_tables_rejects_duplicate_cells() -> None:
    metadata = pd.DataFrame({"cell_id": ["a", "a"], "cycle_life": [100, 101]})
    cycle_summary = pd.DataFrame(
        {"cell_id": ["a"], "cycle_index": [1], "discharge_capacity": [1.0]}
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_processed_tables(metadata, cycle_summary)


def test_validate_processed_tables_rejects_zero_indexed_cycles() -> None:
    metadata = pd.DataFrame({"cell_id": ["a"], "cycle_life": [100]})
    cycle_summary = pd.DataFrame(
        {"cell_id": ["a"], "cycle_index": [0], "discharge_capacity": [1.0]}
    )
    with pytest.raises(ValueError, match="one-indexed"):
        validate_processed_tables(metadata, cycle_summary)


def test_validate_split_assignments_rejects_unlabeled_training_rows() -> None:
    metadata = pd.DataFrame({"cell_id": ["a", "b"], "cycle_life": [100, None]})
    splits = pd.DataFrame({"cell_id": ["a", "b"], "split": ["train", "train"]})
    with pytest.raises(ValueError, match="without cycle_life"):
        validate_split_assignments(metadata, splits)
