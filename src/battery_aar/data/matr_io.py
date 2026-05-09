"""Best-effort loader for MatR MATLAB/HDF5 battery batch structs.

The MatR battery datasets are distributed as MATLAB structs. Newer `.mat` files
are HDF5-backed and can be loaded with h5py; older files may require scipy.io
(not implemented here). This module extracts a normalized cell metadata table
and a cycle summary table. It raises readable errors instead of silently
inventing data when a file has an unexpected layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from battery_aar.data.schema import CELL_METADATA_COLUMNS, CYCLE_SUMMARY_COLUMNS, ensure_columns
from battery_aar.protocols.policy_space import compute_cc4

SUMMARY_FIELD_MAP = {
    "cycle_index": ("cycle", "cycle_index", "Cycle", "cycle_number"),
    "discharge_capacity": ("QDischarge", "q_discharge", "discharge_capacity", "QD"),
    "charge_capacity": ("QCharge", "q_charge", "charge_capacity", "QC"),
    "internal_resistance": ("IR", "internal_resistance", "InternalResistance"),
    "temperature_max": ("Tmax", "temperature_max", "temp_max"),
    "temperature_mean": ("Tavg", "temperature_mean", "Tmean", "temp_mean"),
    "temperature_min": ("Tmin", "temperature_min", "temp_min"),
    "charge_time": ("chargetime", "charge_time", "chargeTime"),
}

METADATA_FIELD_CANDIDATES = {
    "cycle_life": ("cycle_life", "cyclelife", "lifetime"),
    "policy_readable": ("policy_readable", "policy", "protocol", "protocol_readable"),
}


@dataclass
class LoadedBatch:
    """Normalized output from one or more raw batch files."""

    metadata: pd.DataFrame
    cycle_summary: pd.DataFrame


def find_raw_batch_files(raw_dir: str | Path) -> list[Path]:
    """Find likely MatR raw files below `raw_dir`."""

    raw_dir = Path(raw_dir)
    patterns = ["*.mat", "*.h5", "*.hdf5", "*.csv"]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(raw_dir.rglob(pattern))
    return sorted(files)


def load_raw_batches(
    raw_dir: str | Path,
    *,
    max_cells_per_batch: int | None = None,
    first_n_cycles: int | None = 100,
) -> LoadedBatch:
    """Load all recognized raw batch files under `raw_dir` into normalized tables."""

    files = find_raw_batch_files(raw_dir)
    if not files:
        raise FileNotFoundError(
            f"No .mat/.h5/.hdf5/.csv files found under {raw_dir}. "
            "Download the public MatR batches and place them there first."
        )

    metadata_parts: list[pd.DataFrame] = []
    summary_parts: list[pd.DataFrame] = []
    errors: list[str] = []
    for file_path in files:
        try:
            if file_path.suffix.lower() == ".csv":
                loaded = load_csv_cell_or_summary(file_path, first_n_cycles=first_n_cycles)
            else:
                loaded = load_hdf5_mat_batch(
                    file_path,
                    max_cells=max_cells_per_batch,
                    first_n_cycles=first_n_cycles,
                )
            if not loaded.metadata.empty:
                metadata_parts.append(loaded.metadata)
            if not loaded.cycle_summary.empty:
                summary_parts.append(loaded.cycle_summary)
        except Exception as exc:  # message is surfaced to user
            errors.append(f"{file_path}: {exc}")

    if not metadata_parts and not summary_parts:
        details = "\n".join(errors[:10])
        raise RuntimeError(f"Could not parse any raw batch files. First errors:\n{details}")

    metadata = pd.concat(metadata_parts, ignore_index=True) if metadata_parts else pd.DataFrame()
    cycle_summary = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()
    return LoadedBatch(
        metadata=ensure_columns(metadata, CELL_METADATA_COLUMNS),
        cycle_summary=ensure_columns(cycle_summary, CYCLE_SUMMARY_COLUMNS),
    )


def load_csv_cell_or_summary(path: str | Path, *, first_n_cycles: int | None = 100) -> LoadedBatch:
    """Load already-tabular CSV data if raw MatR CSVs are downloaded.

    Supports two lightweight cases:
    1. CSV already contains `cell_id`, `cycle_life`, and optional protocol columns.
    2. CSV contains cycle summary rows with `cell_id`, `cycle_index`, and capacity.
    """

    path = Path(path)
    df = pd.read_csv(path)
    lower = {c.lower(): c for c in df.columns}

    metadata = pd.DataFrame()
    summary = pd.DataFrame()
    if {"cell_id", "cycle_life"}.issubset(lower):
        metadata = df.rename(columns={lower["cell_id"]: "cell_id", lower["cycle_life"]: "cycle_life"})
        if "batch_id" not in metadata.columns:
            metadata["batch_id"] = path.parent.name
        metadata = _parse_protocol_columns(metadata)

    if {"cell_id", "cycle_index"}.issubset(lower) or {"cell_id", "cycle"}.issubset(lower):
        rename: dict[str, str] = {}
        for canonical, candidates in SUMMARY_FIELD_MAP.items():
            for candidate in candidates:
                if candidate.lower() in lower:
                    rename[lower[candidate.lower()]] = canonical
                    break
        summary = df.rename(columns=rename)
        if "cycle" in summary.columns and "cycle_index" not in summary.columns:
            summary = summary.rename(columns={"cycle": "cycle_index"})
        if "batch_id" not in summary.columns:
            summary["batch_id"] = path.parent.name
        if first_n_cycles is not None and "cycle_index" in summary.columns:
            summary = summary[pd.to_numeric(summary["cycle_index"], errors="coerce") <= first_n_cycles]

    return LoadedBatch(
        metadata=ensure_columns(metadata, CELL_METADATA_COLUMNS),
        cycle_summary=ensure_columns(summary, CYCLE_SUMMARY_COLUMNS),
    )


def load_hdf5_mat_batch(
    path: str | Path,
    *,
    max_cells: int | None = None,
    first_n_cycles: int | None = 100,
) -> LoadedBatch:
    """Load an HDF5-backed MATLAB batch struct into normalized tables."""

    path = Path(path)
    with h5py.File(path, "r") as handle:
        batch = _find_batch_group(handle)
        n_cells = _infer_n_cells(batch)
        if max_cells is not None:
            n_cells = min(n_cells, max_cells)

        metadata_rows: list[dict[str, Any]] = []
        summary_rows: list[dict[str, Any]] = []
        for idx in range(n_cells):
            cell_id = _safe_cell_field_string(handle, batch, idx, "cell_id") or f"{path.stem}_cell_{idx:03d}"
            batch_id = path.stem
            cycle_life = _first_present_scalar(handle, batch, idx, METADATA_FIELD_CANDIDATES["cycle_life"])
            policy_readable = _first_present_string(handle, batch, idx, METADATA_FIELD_CANDIDATES["policy_readable"])
            policy_values = _safe_cell_field_vector(handle, batch, idx, "policy")
            cc1, cc2, cc3, cc4 = _protocol_from_values_or_string(policy_values, policy_readable)

            metadata_rows.append(
                {
                    "cell_id": cell_id,
                    "batch_id": batch_id,
                    "cycle_life": cycle_life,
                    "protocol_readable": policy_readable,
                    "cc1": cc1,
                    "cc2": cc2,
                    "cc3": cc3,
                    "cc4": cc4,
                }
            )

            summary_group = _safe_cell_group(handle, batch, idx, "summary")
            if summary_group is None:
                summary_group = _safe_cell_group(handle, batch, idx, "cycles")
            if summary_group is not None:
                summary_df = _summary_group_to_frame(handle, summary_group)
                if first_n_cycles is not None and "cycle_index" in summary_df.columns:
                    summary_df = summary_df[
                        pd.to_numeric(summary_df["cycle_index"], errors="coerce") <= first_n_cycles
                    ]
                for _, row in summary_df.iterrows():
                    item = {"cell_id": cell_id, "batch_id": batch_id}
                    item.update(row.to_dict())
                    summary_rows.append(item)

    return LoadedBatch(
        metadata=ensure_columns(pd.DataFrame(metadata_rows), CELL_METADATA_COLUMNS),
        cycle_summary=ensure_columns(pd.DataFrame(summary_rows), CYCLE_SUMMARY_COLUMNS),
    )


def _find_batch_group(handle: h5py.File) -> h5py.Group:
    if "batch" in handle and isinstance(handle["batch"], h5py.Group):
        return handle["batch"]
    # Some synthetic or exported files use a top-level group with a batch-like shape.
    group_names = [k for k in handle.keys() if isinstance(handle[k], h5py.Group)]
    if len(group_names) == 1:
        return handle[group_names[0]]
    raise ValueError("expected an HDF5 group named 'batch' or a single top-level group")


def _infer_n_cells(batch: h5py.Group) -> int:
    for field in ("cycle_life", "cyclelife", "summary", "cycles", "policy", "cell_id"):
        if field in batch:
            shape = batch[field].shape
            return int(max(shape)) if shape else 1
    raise ValueError("could not infer number of cells from batch fields")


def _cell_ref(batch: h5py.Group, idx: int, field: str) -> Any:
    dataset = batch[field]
    if dataset.shape == ():
        return dataset[()]
    if len(dataset.shape) == 1:
        return dataset[idx]
    if dataset.shape[0] == 1:
        return dataset[0, idx]
    return dataset[idx, 0]


def _safe_cell_group(handle: h5py.File, batch: h5py.Group, idx: int, field: str) -> h5py.Group | None:
    if field not in batch:
        return None
    try:
        ref = _cell_ref(batch, idx, field)
        node = handle[ref] if isinstance(ref, h5py.Reference) else batch[field]
    except Exception:
        return None
    return node if isinstance(node, h5py.Group) else None


def _first_present_scalar(
    handle: h5py.File, batch: h5py.Group, idx: int, fields: tuple[str, ...]
) -> float | None:
    for field in fields:
        value = _safe_cell_field_scalar(handle, batch, idx, field)
        if value is not None:
            return value
    return None


def _first_present_string(
    handle: h5py.File, batch: h5py.Group, idx: int, fields: tuple[str, ...]
) -> str | None:
    for field in fields:
        value = _safe_cell_field_string(handle, batch, idx, field)
        if value:
            return value
    return None


def _safe_cell_field_scalar(handle: h5py.File, batch: h5py.Group, idx: int, field: str) -> float | None:
    values = _safe_cell_field_vector(handle, batch, idx, field)
    if values.size == 0:
        return None
    try:
        value = float(np.ravel(values)[0])
        return value if np.isfinite(value) else None
    except Exception:
        return None


def _safe_cell_field_string(handle: h5py.File, batch: h5py.Group, idx: int, field: str) -> str | None:
    if field not in batch:
        return None
    try:
        ref = _cell_ref(batch, idx, field)
        if isinstance(ref, h5py.Reference):
            arr = np.asarray(handle[ref][()])
        else:
            arr = np.asarray(ref)
        return _matlab_string(arr)
    except Exception:
        return None


def _safe_cell_field_vector(handle: h5py.File, batch: h5py.Group, idx: int, field: str) -> np.ndarray:
    if field not in batch:
        return np.asarray([])
    try:
        ref = _cell_ref(batch, idx, field)
        node = handle[ref] if isinstance(ref, h5py.Reference) else np.asarray(ref)
        return _read_numeric_node(handle, node)
    except Exception:
        return np.asarray([])


def _read_numeric_node(handle: h5py.File, node_or_dataset: h5py.Dataset | np.ndarray) -> np.ndarray:
    arr = node_or_dataset[()] if isinstance(node_or_dataset, h5py.Dataset) else np.asarray(node_or_dataset)
    arr = np.asarray(arr)
    if arr.dtype == h5py.ref_dtype or arr.dtype.kind == "O":
        chunks = []
        for ref in arr.ravel():
            if isinstance(ref, h5py.Reference):
                try:
                    chunks.append(np.ravel(np.asarray(handle[ref][()])))
                except Exception:
                    continue
        if not chunks:
            return np.asarray([])
        return np.concatenate(chunks).astype(float, copy=False)
    if arr.dtype.kind in {"S", "U"}:
        return np.asarray([])
    return np.ravel(arr).astype(float, copy=False)


def _matlab_string(arr: np.ndarray) -> str | None:
    arr = np.asarray(arr).squeeze()
    if arr.size == 0:
        return None
    if arr.dtype.kind in {"S", "U"}:
        if arr.shape == ():
            item = arr.item()
            return (item.decode() if isinstance(item, bytes) else str(item)).strip()
        return "".join(x.decode() if isinstance(x, bytes) else str(x) for x in arr.ravel()).strip()
    if arr.dtype.kind in {"u", "i", "f"}:
        chars = [chr(int(x)) for x in arr.ravel() if int(x) != 0]
        text = "".join(chars).strip()
        return text or None
    return None


def _summary_group_to_frame(handle: h5py.File, group: h5py.Group) -> pd.DataFrame:
    data: dict[str, np.ndarray] = {}
    for canonical, candidates in SUMMARY_FIELD_MAP.items():
        for candidate in candidates:
            if candidate in group:
                data[canonical] = _read_numeric_node(handle, group[candidate])
                break

    if not data:
        return pd.DataFrame(columns=CYCLE_SUMMARY_COLUMNS)

    max_len = max(len(v) for v in data.values() if len(v) > 0)
    normalized: dict[str, np.ndarray] = {}
    for key, values in data.items():
        if len(values) == max_len:
            normalized[key] = values
        elif len(values) == 1:
            normalized[key] = np.repeat(values[0], max_len)
        else:
            padded = np.full(max_len, np.nan)
            padded[: min(max_len, len(values))] = values[:max_len]
            normalized[key] = padded

    frame = pd.DataFrame(normalized)
    if "cycle_index" not in frame.columns:
        frame.insert(0, "cycle_index", np.arange(1, len(frame) + 1))
    return frame


def _protocol_from_values_or_string(values: np.ndarray, readable: str | None) -> tuple[float | None, ...]:
    # Prefer a readable protocol string when present. In MATLAB structs, the
    # `policy` field can be a reference to a char array; reading it numerically
    # yields ASCII/UTF-16 codepoints, not C-rates.
    if readable:
        import re

        nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", readable)]
        if len(nums) >= 3:
            cc1, cc2, cc3 = nums[:3]
            cc4 = nums[3] if len(nums) >= 4 else compute_cc4(cc1, cc2, cc3)
            return cc1, cc2, cc3, cc4
    if values.size >= 3:
        cc1, cc2, cc3 = [float(x) for x in np.ravel(values)[:3]]
        cc4 = float(np.ravel(values)[3]) if values.size >= 4 else compute_cc4(cc1, cc2, cc3)
        return cc1, cc2, cc3, cc4
    return None, None, None, None


def _parse_protocol_columns(metadata: pd.DataFrame) -> pd.DataFrame:
    metadata = metadata.copy()
    lower = {c.lower(): c for c in metadata.columns}
    for canonical in ("cc1", "cc2", "cc3", "cc4", "protocol_readable", "batch_id"):
        if canonical not in metadata.columns and canonical.lower() in lower:
            metadata = metadata.rename(columns={lower[canonical.lower()]: canonical})
    if {"cc1", "cc2", "cc3"}.issubset(metadata.columns) and "cc4" not in metadata.columns:
        metadata["cc4"] = [
            compute_cc4(float(a), float(b), float(c))
            for a, b, c in metadata[["cc1", "cc2", "cc3"]].to_numpy()
        ]
    if "protocol_readable" not in metadata.columns and {"cc1", "cc2", "cc3", "cc4"}.issubset(
        metadata.columns
    ):
        metadata["protocol_readable"] = metadata.apply(
            lambda r: f"{float(r.cc1):.1f}C-{float(r.cc2):.1f}C-{float(r.cc3):.1f}C-{float(r.cc4):.3f}C",
            axis=1,
        )
    return metadata
