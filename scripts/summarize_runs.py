#!/usr/bin/env python
"""Summarize local baseline and candidate run directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from battery_aar.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = summarize_runs(args.runs_dir, args.reports_dir)
    print(_markdown_table(pd.DataFrame(rows)))


def summarize_runs(runs_dir: Path, reports_dir: Path) -> list[dict[str, Any]]:
    """Scan `runs_dir` and write CSV/Markdown summaries under `reports_dir`."""

    reports_dir = ensure_dir(reports_dir)
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    if not runs_dir.exists():
        warnings.append(f"runs directory does not exist: {runs_dir}")
    else:
        run_dirs = sorted(path for path in runs_dir.rglob("*") if path.is_dir())
        leaf_dirs = [path for path in run_dirs if _looks_like_run_dir(path)]
        if not leaf_dirs:
            warnings.append(f"no run artifact directories found under {runs_dir}")
        for run_dir in leaf_dirs:
            run_rows, run_warnings = _summarize_one_run(run_dir)
            rows.extend(run_rows)
            warnings.extend(run_warnings)

    summary = pd.DataFrame(rows)
    summary.to_csv(reports_dir / "run_summary.csv", index=False)
    body = "# Run Summary\n\n"
    if warnings:
        body += "## Warnings\n\n" + "\n".join(f"- {warning}" for warning in warnings) + "\n\n"
    body += _markdown_table(summary) + "\n"
    (reports_dir / "run_summary.md").write_text(body)
    return rows


def _looks_like_run_dir(path: Path) -> bool:
    artifact_names = {
        "metrics.json",
        "model_card.json",
        "predictions.csv",
        "predictions_all_splits.csv",
        "candidate_result.json",
    }
    return any((path / name).exists() for name in artifact_names)


def _summarize_one_run(run_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    metrics_path = run_dir / "metrics.json"
    model_card_path = run_dir / "model_card.json"
    candidate_result_path = run_dir / "candidate_result.json"
    predictions_path = run_dir / "predictions_all_splits.csv"
    if not predictions_path.exists():
        predictions_path = run_dir / "predictions.csv"

    metrics = _read_json_or_warn(metrics_path, warnings)
    model_card = _read_json_or_warn(model_card_path, warnings)
    candidate_result = _read_json_or_warn(candidate_result_path, warnings, required=False)
    n_predictions = _count_predictions(predictions_path, warnings)

    if not metrics and candidate_result.get("metrics"):
        metrics = candidate_result["metrics"]

    rows: list[dict[str, Any]] = []
    if metrics and all(isinstance(v, dict) for v in metrics.values()):
        for split, split_metrics in sorted(metrics.items()):
            rows.append(
                _run_row(run_dir, split, split_metrics, model_card, candidate_result, n_predictions)
            )
    elif metrics:
        rows.append(_run_row(run_dir, "score", metrics, model_card, candidate_result, n_predictions))
    else:
        warnings.append(f"{run_dir}: missing metrics.json and candidate_result metrics")
        rows.append(_run_row(run_dir, "", {}, model_card, candidate_result, n_predictions))
    return rows, warnings


def _read_json_or_warn(
    path: Path,
    warnings: list[str],
    *,
    required: bool = True,
) -> dict[str, Any]:
    if not path.exists():
        if required:
            warnings.append(f"{path.parent}: missing {path.name}")
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        warnings.append(f"{path}: invalid JSON: {exc}")
        return {}


def _count_predictions(path: Path, warnings: list[str]) -> int:
    if not path.exists():
        warnings.append(f"{path.parent}: missing predictions artifact")
        return 0
    try:
        return int(len(pd.read_csv(path)))
    except Exception as exc:
        warnings.append(f"{path}: could not read predictions: {exc}")
        return 0


def _run_row(
    run_dir: Path,
    split: str,
    metrics: dict[str, Any],
    model_card: dict[str, Any],
    candidate_result: dict[str, Any],
    n_predictions: int,
) -> dict[str, Any]:
    return {
        "run_name": run_dir.name,
        "output_dir": str(run_dir),
        "model_kind": model_card.get("model_kind", candidate_result.get("model_kind", "")),
        "max_cycle": model_card.get("max_cycle", candidate_result.get("max_cycle", "")),
        "seed": model_card.get("seed", candidate_result.get("seed", "")),
        "split": split,
        "rmse": metrics.get("rmse", np.nan),
        "mae": metrics.get("mae", np.nan),
        "r2": metrics.get("r2", np.nan),
        "n": metrics.get("n", np.nan),
        "n_predictions": n_predictions,
    }


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No runs found._"
    display = df.copy()
    for col in ("rmse", "mae", "r2"):
        if col in display:
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
    display = display.fillna("")
    headers = [str(c) for c in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        values = [str(row[c]).replace("\n", " ") for c in display.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
