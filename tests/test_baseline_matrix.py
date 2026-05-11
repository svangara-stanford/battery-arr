from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def _write_tiny_raw(raw_dir: Path, *, n_batches: int = 3) -> None:
    raw_dir.mkdir(parents=True)
    cells = []
    summaries = []
    for i in range(6):
        batch = f"b{i % n_batches}"
        protocol = f"{4.0 + i * 0.1:.1f}C-6.0C-5.0C-4.0C"
        cell_id = f"cell_{i}"
        cells.append(
            {
                "cell_id": cell_id,
                "batch_id": batch,
                "cycle_life": 800 + 10 * i,
                "protocol_readable": protocol,
                "cc1": 4.0 + i * 0.1,
                "cc2": 6.0,
                "cc3": 5.0,
                "cc4": 4.0,
            }
        )
        for cycle in range(1, 6):
            summaries.append(
                {
                    "cell_id": cell_id,
                    "batch_id": batch,
                    "cycle_index": cycle,
                    "discharge_capacity": 1.1 - 0.001 * cycle - 0.0001 * i,
                    "charge_capacity": 1.11 - 0.001 * cycle,
                    "internal_resistance": 0.017 + 0.0001 * cycle,
                    "charge_time": 10.0 + 0.01 * cycle,
                }
            )
    pd.DataFrame(cells).to_csv(raw_dir / "metadata.csv", index=False)
    pd.DataFrame(summaries).to_csv(raw_dir / "cycle_summary.csv", index=False)


def test_real_data_baseline_matrix_generates_summary(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    raw = tmp_path / "raw"
    _write_tiny_raw(raw)
    subprocess.run(
        [
            sys.executable,
            str(repo / "scripts/run_real_data_baseline_matrix.py"),
            "--raw-dir",
            str(raw),
            "--processed-dir",
            str(tmp_path / "processed"),
            "--out-root",
            str(tmp_path / "runs"),
            "--reports-dir",
            str(tmp_path / "reports"),
            "--model-kinds",
            "dummy_mean",
            "--split-modes",
            "random",
            "--max-cycle",
            "5",
            "--first-n-cycles",
            "5",
            "--max-cells-per-batch",
            "0",
        ],
        cwd=repo,
        check=True,
    )
    summary = pd.read_csv(tmp_path / "reports/real_data_baseline_matrix.csv")
    assert summary.loc[0, "status"] == "ok"
    assert summary.loc[0, "n_train"] > 0
    assert (tmp_path / "reports/real_data_baseline_matrix.md").exists()


def test_real_data_baseline_matrix_records_split_failure(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    raw = tmp_path / "raw"
    _write_tiny_raw(raw, n_batches=1)
    subprocess.run(
        [
            sys.executable,
            str(repo / "scripts/run_real_data_baseline_matrix.py"),
            "--raw-dir",
            str(raw),
            "--processed-dir",
            str(tmp_path / "processed"),
            "--out-root",
            str(tmp_path / "runs"),
            "--reports-dir",
            str(tmp_path / "reports"),
            "--model-kinds",
            "dummy_mean",
            "--split-modes",
            "batch",
            "--max-cycle",
            "5",
            "--first-n-cycles",
            "5",
            "--max-cells-per-batch",
            "0",
        ],
        cwd=repo,
        check=True,
    )
    summary = pd.read_csv(tmp_path / "reports/real_data_baseline_matrix.csv")
    assert summary.loc[0, "status"] == "failed"
    assert "requires at least 3 labeled batches" in summary.loc[0, "failure_reason"]


def test_smoke_matrix_report_names_and_warnings(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    raw = tmp_path / "raw"
    _write_tiny_raw(raw)
    subprocess.run(
        [
            sys.executable,
            str(repo / "scripts/run_real_data_baseline_matrix.py"),
            "--raw-dir",
            str(raw),
            "--processed-dir",
            str(tmp_path / "processed"),
            "--out-root",
            str(tmp_path / "runs"),
            "--reports-dir",
            str(tmp_path / "reports"),
            "--model-kinds",
            "dummy_mean",
            "--split-modes",
            "random",
            "--max-cycle",
            "5",
            "--first-n-cycles",
            "5",
            "--max-cells-per-batch",
            "2",
            "--smoke",
        ],
        cwd=repo,
        check=True,
    )
    assert (tmp_path / "reports/real_data_baseline_matrix_smoke.csv").exists()
    md = (tmp_path / "reports/real_data_baseline_matrix_smoke.md").read_text()
    assert "smoke/capped" in md
    assert "n_test < 5" in md
    coverage_path = tmp_path / "reports/dataset_coverage_smoke.csv"
    assert coverage_path.exists()
    coverage = pd.read_csv(coverage_path)
    expected_columns = {
        "raw_files_discovered",
        "processed_cells",
        "labeled_cells",
        "unlabeled_cells",
        "labeled_batches",
        "cells_per_batch",
        "labeled_cells_per_batch",
        "protocols_per_batch",
        "cycle_life_missing",
        "cycle_index_min",
        "cycle_index_max",
        "cycle_index_count",
        "cells_fewer_than_100_cycles",
        "run_mode",
        "first_n_cycles",
        "max_cells_per_batch",
    }
    assert expected_columns.issubset(coverage.columns)


def test_default_matrix_includes_paper_ridge_loglife(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    raw = tmp_path / "raw"
    _write_tiny_raw(raw)
    subprocess.run(
        [
            sys.executable,
            str(repo / "scripts/run_real_data_baseline_matrix.py"),
            "--raw-dir",
            str(raw),
            "--processed-dir",
            str(tmp_path / "processed"),
            "--out-root",
            str(tmp_path / "runs"),
            "--reports-dir",
            str(tmp_path / "reports"),
            "--split-modes",
            "random",
            "--max-cycle",
            "5",
            "--first-n-cycles",
            "5",
            "--max-cells-per-batch",
            "0",
        ],
        cwd=repo,
        check=True,
    )
    summary = pd.read_csv(tmp_path / "reports/real_data_baseline_matrix.csv")
    assert "paper_ridge_loglife" in set(summary["model_kind"])
