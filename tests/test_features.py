from __future__ import annotations

import pandas as pd

from battery_aar.features.early_cycle import build_early_cycle_features


def test_build_early_cycle_features_basic() -> None:
    metadata = pd.DataFrame(
        {
            "cell_id": ["a"],
            "batch_id": ["b1"],
            "cycle_life": [1000],
            "protocol_readable": ["3.6C-6.0C-5.6C-4.0C"],
            "cc1": [3.6],
            "cc2": [6.0],
            "cc3": [5.6],
            "cc4": [4.0],
        }
    )
    cycle_summary = pd.DataFrame(
        {
            "cell_id": ["a"] * 5,
            "batch_id": ["b1"] * 5,
            "cycle_index": [1, 2, 3, 4, 5],
            "discharge_capacity": [1.10, 1.09, 1.08, 1.07, 1.06],
            "charge_capacity": [1.11, 1.10, 1.09, 1.08, 1.07],
            "internal_resistance": [0.017, 0.018, 0.019, 0.020, 0.021],
        }
    )
    features = build_early_cycle_features(metadata, cycle_summary, max_cycle=5)
    assert len(features) == 1
    assert features.loc[0, "qd_cycle_2"] == 1.09
    assert features.loc[0, "qd_slope_all"] < 0
    assert "protocol_rms_c_rate" in features.columns
    assert features.loc[0, "cycle_life"] == 1000
