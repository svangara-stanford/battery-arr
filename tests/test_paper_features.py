from __future__ import annotations

import json

import numpy as np
import pandas as pd

from battery_aar.features.paper_features import build_paper_features
from battery_aar.models.baseline import predict, predict_log10_cycle_life, train_baseline


def _metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cell_id": ["a"],
            "batch_id": ["b1"],
            "cycle_life": [1000.0],
            "protocol_readable": ["4.0C-6.0C-5.0C-4.0C"],
            "cc1": [4.0],
            "cc2": [6.0],
            "cc3": [5.0],
            "cc4": [4.0],
        }
    )


def test_paper_features_scalar_slopes_and_differences() -> None:
    cycles = np.arange(1, 101)
    qd = 1.2 - 0.002 * cycles
    energy = 4.0 - 0.01 * cycles
    cycle_summary = pd.DataFrame(
        {
            "cell_id": ["a"] * len(cycles),
            "batch_id": ["b1"] * len(cycles),
            "cycle_index": cycles,
            "discharge_capacity": qd,
            "discharge_energy": energy,
        }
    )

    features = build_paper_features(_metadata(), cycle_summary)
    row = features.iloc[0]
    assert row["paper_qd_cycle_2"] == 1.196
    assert row["paper_qd_cycle_100"] == 1.0
    assert np.isclose(row["paper_qd_linear_slope_2_100"], -0.002)
    assert np.isclose(row["paper_qd_linear_intercept_2_100"], 1.2)
    assert np.isclose(row["paper_qd_linear_slope_91_100"], -0.002)
    assert np.isclose(row["paper_qd_cycle_100_minus_10"], -0.18)
    assert np.isclose(row["paper_sum_sq_qd_diff_10_100_scalar_approx"], 0.0324)
    assert np.isclose(row["paper_energy_cycle_100_minus_10"], -0.9)
    assert bool(row["paper_curve_features_available"]) is False


def test_paper_features_curve_delta_when_arrays_are_available() -> None:
    cycle_summary = pd.DataFrame(
        {
            "cell_id": ["a", "a"],
            "batch_id": ["b1", "b1"],
            "cycle_index": [10, 100],
            "discharge_capacity": [1.1, 1.0],
            "qdlin": [json.dumps([1.0, 0.9, 0.8]), json.dumps([0.95, 0.85, 0.75])],
        }
    )
    features = build_paper_features(_metadata(), cycle_summary)
    row = features.iloc[0]
    assert bool(row["paper_curve_features_available"]) is True
    assert np.isclose(row["paper_curve_min_delta_q_100_10"], -0.05)
    assert np.isclose(row["paper_curve_sum_delta_q_100_10"], -0.15)


def test_paper_loglife_baseline_predicts_back_transformed_cycle_life() -> None:
    features = pd.DataFrame(
        {
            "cell_id": ["a", "b", "c"],
            "split": ["train", "train", "test"],
            "cycle_life": [100.0, 1000.0, 316.227766],
            "paper_qd_linear_slope_2_100": [-0.01, -0.001, -0.005],
            "paper_qd_cycle_2": [1.0, 1.1, 1.05],
        }
    )
    trained = train_baseline(features, model_kind="paper_ridge_loglife", seed=1)
    y_pred = predict(trained, features)
    y_pred_log = predict_log10_cycle_life(trained, features)
    assert trained.target_transform == "log10_cycle_life"
    assert np.all(y_pred > 0)
    assert np.allclose(y_pred, 10.0**y_pred_log)


def test_train_baseline_drops_all_missing_training_feature() -> None:
    features = pd.DataFrame(
        {
            "cell_id": ["a", "b", "c"],
            "split": ["train", "train", "test"],
            "cycle_life": [100.0, 200.0, 300.0],
            "usable": [1.0, 2.0, 3.0],
            "missing_in_train": [np.nan, np.nan, 1.0],
        }
    )
    trained = train_baseline(features, model_kind="ridge", seed=1)
    assert "missing_in_train" not in trained.feature_columns
    assert trained.dropped_feature_columns == ["missing_in_train"]


def test_train_baseline_records_fully_all_missing_feature() -> None:
    features = pd.DataFrame(
        {
            "cell_id": ["a", "b", "c"],
            "split": ["train", "train", "test"],
            "cycle_life": [100.0, 200.0, 300.0],
            "usable": [1.0, 2.0, 3.0],
            "paper_curve_min_delta_q_100_10": [np.nan, np.nan, np.nan],
        }
    )
    trained = train_baseline(features, model_kind="paper_ridge_loglife", seed=1)
    assert "paper_curve_min_delta_q_100_10" not in trained.feature_columns
    assert trained.dropped_feature_columns == ["paper_curve_min_delta_q_100_10"]
