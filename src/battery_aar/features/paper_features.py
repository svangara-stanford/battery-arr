"""Paper-inspired early-cycle features for cycle-life prediction.

These features follow the structure of the Severson/Attia early prediction
workflow used around the Chueh/Toyota fast-charging studies, but they are not an
exact reproduction unless interpolated Q(V)/V(Q) traces and original trained
feature indices are available. With the current MatR JSON ZIP preprocessing, the
module builds the scalar discharge-capacity and slope subset, marks unavailable
curve-derived quantities, and preserves feature provenance in column names.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

PAPER_DIRECT_FEATURES = [
    "paper_qd_cycle_2",
    "paper_qd_cycle_100",
    "paper_qd_max_capacity_change_2_100",
    "paper_qd_linear_slope_2_100",
    "paper_qd_linear_intercept_2_100",
    "paper_qd_linear_slope_91_100",
    "paper_qd_linear_intercept_91_100",
    "paper_qd_cycle_100_minus_10",
    "paper_abs_qd_cycle_100_minus_10",
    "paper_log_abs_qd_cycle_100_minus_10",
    "paper_sum_qd_diff_10_100_scalar_approx",
    "paper_sum_sq_qd_diff_10_100_scalar_approx",
    "paper_energy_cycle_100_minus_10",
]

PAPER_CURVE_FEATURES = [
    "paper_curve_min_delta_q_100_10",
    "paper_curve_var_delta_q_100_10",
    "paper_curve_mean_delta_q_100_10",
    "paper_curve_sum_delta_q_100_10",
    "paper_curve_sum_sq_delta_q_100_10",
]


def build_paper_features(
    metadata: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    *,
    max_cycle: int = 100,
) -> pd.DataFrame:
    """Build one row per cell using BMS-autoanalysis-inspired early features.

    Direct scalar analogs are computed from per-cycle `discharge_capacity` and
    optional `discharge_energy`. Curve-derived Qdlin features are populated only
    when cycle-summary rows include parseable interpolated arrays for cycles 10
    and 100. Otherwise the curve columns are present as NaN and
    `paper_curve_features_available` is false.
    """

    _validate_inputs(metadata, cycle_summary)
    summary = cycle_summary[
        pd.to_numeric(cycle_summary["cycle_index"], errors="coerce") <= max_cycle
    ].copy()
    if summary.empty:
        raise ValueError("No cycle_summary rows remain after max_cycle filtering")

    rows: list[dict[str, Any]] = []
    for cell_id, group in summary.groupby("cell_id", sort=True):
        group = _prepared_group(group)
        row: dict[str, Any] = {"cell_id": cell_id}
        row.update(_scalar_capacity_features(group))
        row.update(_curve_delta_features(group))
        row["paper_n_cycles_used"] = int(len(group))
        rows.append(row)

    features = pd.DataFrame(rows)
    keep = [
        "cell_id",
        *[
            col
            for col in ["cycle_life", "batch_id", "protocol_readable", "cc1", "cc2", "cc3", "cc4"]
            if col in metadata.columns
        ],
    ]
    return features.merge(metadata[keep].drop_duplicates("cell_id"), on="cell_id", how="left")


def paper_feature_documentation() -> dict[str, str]:
    """Return concise provenance notes for the implemented paper features."""

    return {
        "paper_qd_cycle_2": "Direct scalar analog of QDischarge at cycle 2.",
        "paper_qd_cycle_100": "Direct scalar analog of QDischarge at cycle 100.",
        "paper_qd_max_capacity_change_2_100": "Scalar capacity range through cycles 2-100; paper-inspired summary, not Q(V).",
        "paper_qd_linear_slope_2_100": "Linear fit slope of QDischarge over cycles 2-100.",
        "paper_qd_linear_slope_91_100": "Linear fit slope of QDischarge over cycles 91-100.",
        "paper_qd_cycle_100_minus_10": "Scalar capacity difference between cycles 100 and 10.",
        "paper_log_abs_qd_cycle_100_minus_10": "Log absolute scalar capacity difference.",
        "paper_sum_qd_diff_10_100_scalar_approx": "Scalar approximation to Qdlin difference sum when Q(V) arrays are absent.",
        "paper_energy_cycle_100_minus_10": "Energy difference between cycles 100 and 10 when discharge_energy is available.",
        "paper_curve_*": "Direct Qdlin-style curve features only when interpolated per-cycle arrays are available.",
    }


def _validate_inputs(metadata: pd.DataFrame, cycle_summary: pd.DataFrame) -> None:
    missing_meta = {"cell_id"}.difference(metadata.columns)
    missing_summary = {"cell_id", "cycle_index", "discharge_capacity"}.difference(
        cycle_summary.columns
    )
    if missing_meta:
        raise ValueError(f"metadata missing required columns: {sorted(missing_meta)}")
    if missing_summary:
        raise ValueError(f"cycle_summary missing required columns: {sorted(missing_summary)}")


def _prepared_group(group: pd.DataFrame) -> pd.DataFrame:
    out = group.copy()
    out["cycle_index"] = pd.to_numeric(out["cycle_index"], errors="coerce")
    out["discharge_capacity"] = pd.to_numeric(out["discharge_capacity"], errors="coerce")
    if "discharge_energy" in out.columns:
        out["discharge_energy"] = pd.to_numeric(out["discharge_energy"], errors="coerce")
    return out.sort_values("cycle_index")


def _scalar_capacity_features(group: pd.DataFrame) -> dict[str, float]:
    q2 = _value_at_cycle(group, "discharge_capacity", 2)
    q10 = _value_at_cycle(group, "discharge_capacity", 10)
    q100 = _value_at_cycle(group, "discharge_capacity", 100)
    qd_2_100 = group.loc[
        group["cycle_index"].between(2, 100),
        "discharge_capacity",
    ].dropna()
    delta_100_10 = _sub_nan(q100, q10)
    slope_2_100, intercept_2_100 = _linear_fit(group, "discharge_capacity", 2, 100)
    slope_91_100, intercept_91_100 = _linear_fit(group, "discharge_capacity", 91, 100)
    energy_10 = _value_at_cycle(group, "discharge_energy", 10)
    energy_100 = _value_at_cycle(group, "discharge_energy", 100)
    return {
        "paper_qd_cycle_2": q2,
        "paper_qd_cycle_100": q100,
        "paper_qd_max_capacity_change_2_100": _range(qd_2_100),
        "paper_qd_linear_slope_2_100": slope_2_100,
        "paper_qd_linear_intercept_2_100": intercept_2_100,
        "paper_qd_linear_slope_91_100": slope_91_100,
        "paper_qd_linear_intercept_91_100": intercept_91_100,
        "paper_qd_cycle_100_minus_10": delta_100_10,
        "paper_abs_qd_cycle_100_minus_10": _abs(delta_100_10),
        "paper_log_abs_qd_cycle_100_minus_10": _log_abs(delta_100_10),
        "paper_sum_qd_diff_10_100_scalar_approx": delta_100_10,
        "paper_sum_sq_qd_diff_10_100_scalar_approx": _square(delta_100_10),
        "paper_energy_cycle_100_minus_10": _sub_nan(energy_100, energy_10),
    }


def _curve_delta_features(group: pd.DataFrame) -> dict[str, Any]:
    q10 = _curve_at_cycle(group, 10)
    q100 = _curve_at_cycle(group, 100)
    out: dict[str, Any] = {feature: np.nan for feature in PAPER_CURVE_FEATURES}
    out["paper_curve_features_available"] = False
    out["paper_missing_curve_feature_reason"] = "interpolated Qdlin/Vdlin traces not present"
    if q10 is None or q100 is None:
        return out
    n = min(len(q10), len(q100))
    if n == 0:
        return out
    delta = q100[:n] - q10[:n]
    delta = delta[np.isfinite(delta)]
    if delta.size == 0:
        return out
    out.update(
        {
            "paper_curve_min_delta_q_100_10": float(np.min(delta)),
            "paper_curve_var_delta_q_100_10": float(np.var(delta, ddof=0)),
            "paper_curve_mean_delta_q_100_10": float(np.mean(delta)),
            "paper_curve_sum_delta_q_100_10": float(np.sum(delta)),
            "paper_curve_sum_sq_delta_q_100_10": float(np.sum(delta**2)),
            "paper_curve_features_available": True,
            "paper_missing_curve_feature_reason": "",
        }
    )
    return out


def _curve_at_cycle(group: pd.DataFrame, cycle: int) -> np.ndarray | None:
    curve_columns = [
        "qdlin",
        "Qdlin",
        "q_d_lin",
        "discharge_capacity_curve",
        "discharge_capacity_interpolated",
    ]
    row = group.loc[group["cycle_index"] == cycle]
    if row.empty:
        return None
    for col in curve_columns:
        if col in row.columns:
            parsed = _parse_array_like(row.iloc[0][col])
            if parsed is not None:
                return parsed
    return None


def _parse_array_like(value: Any) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        return value.astype(float, copy=False).ravel()
    if isinstance(value, list | tuple):
        return np.asarray(value, dtype=float).ravel()
    if isinstance(value, str) and value.strip():
        try:
            loaded = json.loads(value)
            if isinstance(loaded, Iterable):
                return np.asarray(list(loaded), dtype=float).ravel()
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
    return None


def _value_at_cycle(group: pd.DataFrame, col: str, cycle: int) -> float:
    if col not in group.columns:
        return float("nan")
    values = group.loc[group["cycle_index"] == cycle, col].dropna()
    return float(values.iloc[0]) if not values.empty else float("nan")


def _linear_fit(group: pd.DataFrame, col: str, start: int, end: int) -> tuple[float, float]:
    local = group.loc[group["cycle_index"].between(start, end), ["cycle_index", col]].dropna()
    if len(local) < 2:
        return float("nan"), float("nan")
    x = local["cycle_index"].to_numpy(dtype=float)
    y = local[col].to_numpy(dtype=float)
    x_centered = x - np.mean(x)
    denom = float(np.sum(x_centered**2))
    if denom == 0:
        return float("nan"), float("nan")
    slope = float(np.sum(x_centered * (y - np.mean(y))) / denom)
    intercept = float(np.mean(y) - slope * np.mean(x))
    return slope, intercept


def _range(values: pd.Series) -> float:
    if values.empty:
        return float("nan")
    return float(values.max() - values.min())


def _sub_nan(a: float, b: float) -> float:
    return float(a - b) if np.isfinite(a) and np.isfinite(b) else float("nan")


def _abs(value: float) -> float:
    return float(abs(value)) if np.isfinite(value) else float("nan")


def _square(value: float) -> float:
    return float(value**2) if np.isfinite(value) else float("nan")


def _log_abs(value: float) -> float:
    return float(np.log10(abs(value))) if np.isfinite(value) and value != 0 else float("nan")
