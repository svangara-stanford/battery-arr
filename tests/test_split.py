from __future__ import annotations

import pandas as pd

from battery_aar.data.split import SplitConfig, make_split_assignments


def test_make_split_assignments_is_deterministic() -> None:
    metadata = pd.DataFrame({"cell_id": [f"c{i}" for i in range(20)]})
    s1 = make_split_assignments(metadata, SplitConfig(seed=123))
    s2 = make_split_assignments(metadata, SplitConfig(seed=123))
    assert s1.equals(s2)
    assert set(s1["split"]) == {"train", "val", "test"}


def test_group_split_keeps_groups_together() -> None:
    metadata = pd.DataFrame(
        {
            "cell_id": [f"c{i}" for i in range(12)],
            "batch_id": ["b1"] * 4 + ["b2"] * 4 + ["b3"] * 4,
        }
    )
    split = make_split_assignments(metadata, SplitConfig(seed=1, group_column="batch_id"))
    merged = metadata.merge(split, on="cell_id")
    assert merged.groupby("batch_id")["split"].nunique().max() == 1
