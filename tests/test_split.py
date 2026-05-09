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


def test_named_batch_split_mode_keeps_batches_together() -> None:
    metadata = pd.DataFrame(
        {
            "cell_id": [f"c{i}" for i in range(12)],
            "batch_id": ["b1"] * 4 + ["b2"] * 4 + ["b3"] * 4,
        }
    )
    split = make_split_assignments(metadata, SplitConfig(seed=1, mode="batch"))
    merged = metadata.merge(split, on="cell_id")
    assert merged.groupby("batch_id")["split"].nunique().max() == 1


def test_protocol_and_leave_one_batch_modes() -> None:
    metadata = pd.DataFrame(
        {
            "cell_id": [f"c{i}" for i in range(12)],
            "batch_id": ["b1"] * 4 + ["b2"] * 4 + ["b3"] * 4,
            "protocol_readable": ["p1", "p1", "p2", "p2"] * 3,
            "cc1": [5.6, 5.6, 6.0, 6.0] * 3,
            "cc2": [6.0] * 12,
            "cc3": [4.8] * 12,
            "cc4": [3.574] * 12,
        }
    )
    protocol_split = make_split_assignments(metadata, SplitConfig(seed=2, mode="protocol"))
    merged_protocol = metadata.merge(protocol_split, on="cell_id")
    assert merged_protocol.groupby("protocol_readable")["split"].nunique().max() == 1

    cluster_split = make_split_assignments(metadata, SplitConfig(seed=2, mode="protocol_cluster"))
    merged_cluster = metadata.merge(cluster_split, on="cell_id")
    assert merged_cluster.groupby(["cc1", "cc2", "cc3", "cc4"])["split"].nunique().max() == 1

    lobo = make_split_assignments(
        metadata,
        SplitConfig(mode="leave_one_batch_out", leave_one_group_value="b3", seed=2),
    )
    merged_lobo = metadata.merge(lobo, on="cell_id")
    assert set(merged_lobo.loc[merged_lobo["batch_id"] == "b3", "split"]) == {"test"}
