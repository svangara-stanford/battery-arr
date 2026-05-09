"""Regression metrics for battery lifetime prediction."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return RMSE, MAE, and R² for finite prediction pairs."""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        raise ValueError("no finite y_true/y_pred pairs to score")
    yt = y_true[mask]
    yp = y_pred[mask]
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    r2 = float(r2_score(yt, yp)) if len(yt) >= 2 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2, "n": int(len(yt))}


def battery_pgr(weak_rmse: float, strong_rmse: float, candidate_rmse: float) -> float:
    """Compute a simple Battery-PGR-style gap recovery score.

    Higher is better. 0 means candidate matches weak baseline, 1 means it
    matches strong baseline. Values can be negative or above 1.
    """

    denom = weak_rmse - strong_rmse
    if denom <= 0:
        raise ValueError("weak_rmse must be worse/larger than strong_rmse")
    return float((weak_rmse - candidate_rmse) / denom)
