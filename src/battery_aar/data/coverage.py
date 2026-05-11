"""Dataset coverage reports for processed Battery-AAR tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from battery_aar.data.matr_io import find_raw_batch_files
from battery_aar.utils.io import ensure_dir, write_json


def write_dataset_coverage_reports(
    *,
    metadata: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    reports_dir: str | Path,
    raw_dir: str | Path | None,
    first_n_cycles: int | None,
    max_cells_per_batch: int | None,
    smoke: bool,
    parse_errors: list[str] | None = None,
    parse_errors_unavailable: bool = False,
    suffix: str = "",
) -> tuple[Path, Path]:
    """Write CSV/Markdown coverage reports and return their paths."""

    reports_dir = ensure_dir(reports_dir)
    coverage = dataset_coverage_frame(
        metadata=metadata,
        cycle_summary=cycle_summary,
        raw_dir=raw_dir,
        first_n_cycles=first_n_cycles,
        max_cells_per_batch=max_cells_per_batch,
        smoke=smoke,
    )
    summary = dataset_coverage_summary(
        metadata=metadata,
        cycle_summary=cycle_summary,
        raw_dir=raw_dir,
        first_n_cycles=first_n_cycles,
        max_cells_per_batch=max_cells_per_batch,
        smoke=smoke,
        parse_errors=parse_errors or [],
        parse_errors_unavailable=parse_errors_unavailable,
    )
    stem = f"dataset_coverage{suffix}"
    csv_path = reports_dir / f"{stem}.csv"
    md_path = reports_dir / f"{stem}.md"
    json_path = reports_dir / f"{stem}.json"
    coverage.to_csv(csv_path, index=False)
    write_json(json_path, summary)
    md_path.write_text(_coverage_markdown(coverage, summary))
    return csv_path, md_path


def dataset_coverage_frame(
    *,
    metadata: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    raw_dir: str | Path | None,
    first_n_cycles: int | None,
    max_cells_per_batch: int | None,
    smoke: bool,
) -> pd.DataFrame:
    """Return one coverage row per processed batch."""

    raw_files = find_raw_batch_files(raw_dir) if raw_dir is not None else []
    if metadata.empty:
        return pd.DataFrame(
            [
                _coverage_row(
                    batch_id="ALL",
                    raw_files_discovered=len(raw_files),
                    processed_cells=0,
                    labeled_cells=0,
                    unlabeled_cells=0,
                    labeled_batches=0,
                    cells_per_batch=0,
                    labeled_cells_per_batch=0,
                    protocols_per_batch=0,
                    cycle_life_missing=0,
                    cycle_index_min=pd.NA,
                    cycle_index_max=pd.NA,
                    cycle_index_count=0,
                    cells_fewer_than_100_cycles=0,
                    first_n_cycles=first_n_cycles,
                    max_cells_per_batch=max_cells_per_batch,
                    smoke=smoke,
                )
            ]
        )

    rows: list[dict[str, Any]] = []
    labels = pd.to_numeric(metadata["cycle_life"], errors="coerce")
    labeled_batches = int(metadata.loc[labels.notna(), "batch_id"].nunique())
    cycle_stats = _cycle_stats_by_batch(metadata, cycle_summary)
    for batch_id, batch_meta in metadata.groupby("batch_id", dropna=False):
        batch_labels = pd.to_numeric(batch_meta["cycle_life"], errors="coerce")
        stats = cycle_stats.get(str(batch_id), {})
        rows.append(
            _coverage_row(
                batch_id=str(batch_id),
                raw_files_discovered=len(raw_files),
                processed_cells=int(batch_meta["cell_id"].nunique()),
                labeled_cells=int(batch_labels.notna().sum()),
                unlabeled_cells=int(batch_labels.isna().sum()),
                labeled_batches=labeled_batches,
                cells_per_batch=int(batch_meta["cell_id"].nunique()),
                labeled_cells_per_batch=int(batch_labels.notna().sum()),
                protocols_per_batch=int(batch_meta["protocol_readable"].nunique(dropna=True))
                if "protocol_readable" in batch_meta
                else 0,
                cycle_life_missing=int(batch_labels.isna().sum()),
                cycle_index_min=stats.get("min", pd.NA),
                cycle_index_max=stats.get("max", pd.NA),
                cycle_index_count=stats.get("count", 0),
                cells_fewer_than_100_cycles=stats.get("cells_fewer_than_100", 0),
                first_n_cycles=first_n_cycles,
                max_cells_per_batch=max_cells_per_batch,
                smoke=smoke,
            )
        )
    return pd.DataFrame(rows)


def dataset_coverage_summary(
    *,
    metadata: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    raw_dir: str | Path | None,
    first_n_cycles: int | None,
    max_cells_per_batch: int | None,
    smoke: bool,
    parse_errors: list[str],
    parse_errors_unavailable: bool = False,
) -> dict[str, Any]:
    """Return aggregate dataset coverage metadata."""

    raw_files = find_raw_batch_files(raw_dir) if raw_dir is not None else []
    labels = pd.to_numeric(metadata.get("cycle_life", pd.Series(dtype=float)), errors="coerce")
    labeled = labels.notna()
    per_cell = (
        cycle_summary.groupby("cell_id")["cycle_index"].max()
        if {"cell_id", "cycle_index"}.issubset(cycle_summary.columns)
        else pd.Series(dtype=float)
    )
    return {
        "raw_files_discovered": len(raw_files),
        "raw_files": [str(path) for path in raw_files],
        "processed_cells": int(metadata["cell_id"].nunique()) if "cell_id" in metadata else 0,
        "labeled_cells": int(labeled.sum()),
        "unlabeled_cells": int((~labeled).sum()),
        "labeled_batches": int(metadata.loc[labeled, "batch_id"].nunique())
        if "batch_id" in metadata
        else 0,
        "cycle_life_missing": int((~labeled).sum()),
        "cells_fewer_than_100_cycles": int((per_cell < 100).sum()) if not per_cell.empty else 0,
        "first_n_cycles": first_n_cycles,
        "max_cells_per_batch": max_cells_per_batch,
        "run_mode": "smoke/capped" if smoke else "full/uncapped",
        "parse_error_count": len(parse_errors),
        "parse_errors_unavailable": parse_errors_unavailable,
        "parse_errors_first_20": parse_errors[:20],
    }


def coverage_warnings(summary: dict[str, Any], *, min_test: int | None = None) -> list[str]:
    """Return warnings that prevent interpreting metrics as scientific evidence."""

    warnings: list[str] = []
    if int(summary.get("labeled_cells", 0)) < 20:
        warnings.append("labeled_cells < 20; metrics are smoke-test-only")
    if int(summary.get("labeled_batches", 0)) < 2:
        warnings.append("labeled_batches < 2; batch transfer metrics are not available")
    if min_test is not None and min_test < 5:
        warnings.append("n_test < 5 for at least one successful run; metrics are unstable")
    if int(summary.get("parse_error_count", 0)) > 0:
        warnings.append("some raw members failed to parse; see dataset_coverage.json")
    if bool(summary.get("parse_errors_unavailable", False)):
        warnings.append("parse errors unavailable because preprocessing was skipped")
    return warnings


def _cycle_stats_by_batch(
    metadata: pd.DataFrame,
    cycle_summary: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    if not {"cell_id", "cycle_index"}.issubset(cycle_summary.columns):
        return {}
    merged = cycle_summary[["cell_id", "cycle_index"]].merge(
        metadata[["cell_id", "batch_id"]],
        on="cell_id",
        how="left",
    )
    merged["cycle_index"] = pd.to_numeric(merged["cycle_index"], errors="coerce")
    per_cell = merged.groupby(["batch_id", "cell_id"])["cycle_index"].max().reset_index()
    grouped = merged.groupby("batch_id")["cycle_index"].agg(["min", "max", "count"])
    out: dict[str, dict[str, Any]] = {}
    for batch_id, row in grouped.iterrows():
        batch_cells = per_cell.loc[per_cell["batch_id"] == batch_id]
        out[str(batch_id)] = {
            "min": float(row["min"]),
            "max": float(row["max"]),
            "count": int(row["count"]),
            "cells_fewer_than_100": int((batch_cells["cycle_index"] < 100).sum()),
        }
    return out


def _coverage_row(**kwargs: Any) -> dict[str, Any]:
    return {
        "batch_id": kwargs["batch_id"],
        "raw_files_discovered": kwargs["raw_files_discovered"],
        "processed_cells": kwargs["processed_cells"],
        "labeled_cells": kwargs["labeled_cells"],
        "unlabeled_cells": kwargs["unlabeled_cells"],
        "labeled_batches": kwargs["labeled_batches"],
        "cells_per_batch": kwargs["cells_per_batch"],
        "labeled_cells_per_batch": kwargs["labeled_cells_per_batch"],
        "protocols_per_batch": kwargs["protocols_per_batch"],
        "cycle_life_missing": kwargs["cycle_life_missing"],
        "cycle_index_min": kwargs["cycle_index_min"],
        "cycle_index_max": kwargs["cycle_index_max"],
        "cycle_index_count": kwargs["cycle_index_count"],
        "cells_fewer_than_100_cycles": kwargs["cells_fewer_than_100_cycles"],
        "run_mode": "smoke/capped" if kwargs["smoke"] else "full/uncapped",
        "first_n_cycles": kwargs["first_n_cycles"],
        "max_cells_per_batch": kwargs["max_cells_per_batch"],
    }


def _coverage_markdown(coverage: pd.DataFrame, summary: dict[str, Any]) -> str:
    warnings = coverage_warnings(summary)
    body = "# Dataset Coverage\n\n"
    body += f"Run mode: **{summary['run_mode']}**\n\n"
    body += "## Summary\n\n"
    for key in [
        "raw_files_discovered",
        "processed_cells",
        "labeled_cells",
        "unlabeled_cells",
        "labeled_batches",
        "cycle_life_missing",
        "cells_fewer_than_100_cycles",
        "first_n_cycles",
        "max_cells_per_batch",
        "parse_error_count",
        "parse_errors_unavailable",
    ]:
        body += f"- `{key}`: {summary.get(key)}\n"
    if warnings:
        body += "\n## Warnings\n\n" + "\n".join(f"- {warning}" for warning in warnings) + "\n"
    body += "\n## By Batch\n\n" + _markdown_table(coverage) + "\n"
    return body


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    display = df.fillna("")
    headers = [str(col) for col in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in display.columns) + " |")
    return "\n".join(lines)
