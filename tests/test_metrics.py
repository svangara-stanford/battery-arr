from __future__ import annotations

import numpy as np
import pytest

from battery_aar.evaluation.metrics import battery_pgr, regression_metrics


def test_regression_metrics() -> None:
    metrics = regression_metrics(np.array([1, 2, 3]), np.array([1, 2, 4]))
    assert metrics["rmse"] > 0
    assert metrics["mae"] > 0
    assert metrics["n"] == 3


def test_battery_pgr() -> None:
    assert battery_pgr(100.0, 50.0, 75.0) == 0.5
    with pytest.raises(ValueError):
        battery_pgr(50.0, 100.0, 75.0)
