"""Early-cycle tabular features for battery lifetime prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from battery_aar.protocols.policy_space import protocol_stress_features

PROTOCOL_FEATURES = ["cc1", "cc2", "cc3", "cc4"]
DEFAULT_CAPACITY_CYCLES = (2, 10, 50, 100)


def build_early_cycle_features(
    metadata: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    *,
    max_cycle: int = 100,
    include_protocol: bool = True,
) -> pd.DataFrame:
    """Build one row of early-cycle features per cell.

    The feature set is deliberately interpretable for a skeleton prototype:
    discharge-capacity statistics, specific early-cycle capacity values,
    capacity slopes/curvature proxies, resistance/charge-time/temperature
    summaries, and optional protocol C-rate features.
    """

    _validate_inputs(metadata, cycle_summary)
    summary = cycle_summary[pd.to_numeric(cycle_summary["cycle_index"], errors="coerce") <= max_cycle].copy()
    if summary.empty:
        raise ValueError("No cycle_summary rows remain after max_cycle filtering")

    rows: list[dict[str, float | str]] = []
    for cell_id, group in summary.groupby("cell_id", sort=True):
        group = group.sort_values("cycle_index")
        row: dict[str, float | str] = {"cell_id": cell_id}
        qd = _numeric(group, "discharge_capacity")
        cycles = _numeric(group, "cycle_index")
        row.update(_series_stats(qd, prefix="qd"))
        row.update(_capacity_at_cycles(group, DEFAULT_CAPACITY_CYCLES))
        row["qd_slope_all"] = _safe_slope(cycles, qd)
        row["qd_slope_2_100"] = _window_slope(group, "discharge_capacity", 2, max_cycle)
        row["qd_slope_2_50"] = _window_slope(group, "discharge_capacity", 2, 50)
        row["qd_slope_50_100"] = _window_slope(group, "discharge_capacity", 50, max_cycle)
        row["qd_curvature_proxy"] = _sub_nan(row["qd_slope_50_100"], row["qd_slope_2_50"])
        row["qd_delta_first_last"] = _first_last_delta(qd)
        row["qd_log_abs_delta_first_last"] = _log_abs(row["qd_delta_first_last"])
        early_qd = _value_at_or_after_cycle(group, "discharge_capacity", 2)
        late_qd = _value_at_or_before_cycle(group, "discharge_capacity", max_cycle)
        row["qd_fade_2_to_max_cycle"] = _sub_nan(early_qd, late_qd)
        row["qd_log_abs_fade_2_to_max_cycle"] = _log_abs(row["qd_fade_2_to_max_cycle"])
        row["n_cycles_used"] = int(len(group))

        for col, prefix in [
            ("charge_capacity", "qc"),
            ("internal_resistance", "ir"),
            ("charge_time", "charge_time"),
            ("temperature_max", "tmax"),
            ("temperature_mean", "tmean"),
            ("temperature_min", "tmin"),
        ]:
            if col in group.columns and group[col].notna().any():
                values = _numeric(group, col)
                row.update(_series_stats(values, prefix=prefix))
                row[f"{prefix}_slope_all"] = _safe_slope(cycles, values)
                row[f"{prefix}_delta_first_last"] = _first_last_delta(values)

        rows.append(row)

    features = pd.DataFrame(rows)
    if include_protocol:
        keep = ["cell_id"] + [c for c in PROTOCOL_FEATURES + ["protocol_readable", "batch_id"] if c in metadata]
        features = features.merge(metadata[keep].drop_duplicates("cell_id"), on="cell_id", how="left")
        if set(PROTOCOL_FEATURES).issubset(features.columns):
            stress = protocol_stress_features(features[PROTOCOL_FEATURES])
            features = pd.concat([features, stress], axis=1)
    if "cycle_life" in metadata.columns:
        features = features.merge(metadata[["cell_id", "cycle_life"]], on="cell_id", how="left")
    return features


def _validate_inputs(metadata: pd.DataFrame, cycle_summary: pd.DataFrame) -> None:
    required_meta = {"cell_id"}
    required_summary = {"cell_id", "cycle_index", "discharge_capacity"}
    missing_meta = required_meta.difference(metadata.columns)
    missing_summary = required_summary.difference(cycle_summary.columns)
    if missing_meta:
        raise ValueError(f"metadata missing required columns: {sorted(missing_meta)}")
    if missing_summary:
        raise ValueError(f"cycle_summary missing required columns: {sorted(missing_summary)}")


def _numeric(df: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def _series_stats(values: np.ndarray, *, prefix: str) -> dict[str, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            f"{prefix}_first": np.nan,
            f"{prefix}_last": np.nan,
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
        }
    return {
        f"{prefix}_first": float(values[0]),
        f"{prefix}_last": float(values[-1]),
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values, ddof=0)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
    }


def _capacity_at_cycles(group: pd.DataFrame, cycles: tuple[int, ...]) -> dict[str, float]:
    out: dict[str, float] = {}
    local = group[["cycle_index", "discharge_capacity"]].copy()
    local["cycle_index"] = pd.to_numeric(local["cycle_index"], errors="coerce")
    local["discharge_capacity"] = pd.to_numeric(local["discharge_capacity"], errors="coerce")
    for cycle in cycles:
        vals = local.loc[local["cycle_index"] == cycle, "discharge_capacity"].dropna()
        out[f"qd_cycle_{cycle}"] = float(vals.iloc[0]) if not vals.empty else np.nan
    return out


def _window_slope(group: pd.DataFrame, col: str, start: int, end: int) -> float:
    mask = (pd.to_numeric(group["cycle_index"], errors="coerce") >= start) & (
        pd.to_numeric(group["cycle_index"], errors="coerce") <= end
    )
    if mask.sum() < 2:
        return float("nan")
    return _safe_slope(_numeric(group.loc[mask], "cycle_index"), _numeric(group.loc[mask], col))


def _value_at_or_after_cycle(group: pd.DataFrame, col: str, cycle: int) -> float:
    local = group[["cycle_index", col]].copy()
    local["cycle_index"] = pd.to_numeric(local["cycle_index"], errors="coerce")
    local[col] = pd.to_numeric(local[col], errors="coerce")
    local = local.loc[(local["cycle_index"] >= cycle) & local[col].notna()].sort_values("cycle_index")
    return float(local[col].iloc[0]) if not local.empty else float("nan")


def _value_at_or_before_cycle(group: pd.DataFrame, col: str, cycle: int) -> float:
    local = group[["cycle_index", col]].copy()
    local["cycle_index"] = pd.to_numeric(local["cycle_index"], errors="coerce")
    local[col] = pd.to_numeric(local[col], errors="coerce")
    local = local.loc[(local["cycle_index"] <= cycle) & local[col].notna()].sort_values("cycle_index")
    return float(local[col].iloc[-1]) if not local.empty else float("nan")


def _safe_slope(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    x_centered = x[mask] - np.mean(x[mask])
    denom = float(np.sum(x_centered**2))
    if denom == 0:
        return float("nan")
    return float(np.sum(x_centered * (y[mask] - np.mean(y[mask]))) / denom)


def _first_last_delta(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan")
    return float(values[-1] - values[0])


def _log_abs(value: float) -> float:
    return float(np.log1p(abs(value))) if np.isfinite(value) else float("nan")


def _sub_nan(a: float, b: float) -> float:
    return float(a - b) if np.isfinite(a) and np.isfinite(b) else float("nan")
