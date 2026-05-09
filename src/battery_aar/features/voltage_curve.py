"""Placeholder utilities for future voltage-capacity curve features.

The first proof of concept uses cycle-summary tables. This module is included
so future work has a clear home for voltage-capacity interpolation, dQ/dV, and
dV/dQ features.
"""

from __future__ import annotations

import numpy as np


def interpolate_curve(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Interpolate a finite 1D curve onto a fixed grid."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    grid = np.asarray(grid, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.full_like(grid, np.nan, dtype=float)
    order = np.argsort(x[mask])
    return np.interp(grid, x[mask][order], y[mask][order], left=np.nan, right=np.nan)
