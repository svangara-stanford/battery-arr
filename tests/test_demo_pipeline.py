from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_demo_pipeline_end_to_end(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    demo = tmp_path / "demo"
    run = tmp_path / "run"
    subprocess.run(
        [sys.executable, str(repo / "scripts/generate_demo_data.py"), "--out", str(demo), "--n-cells", "36", "--n-cycles", "50", "--seed", "7"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(repo / "scripts/train_baseline.py"),
            "--processed-dir",
            str(demo),
            "--out",
            str(run),
            "--max-cycle",
            "50",
            "--seed",
            "7",
            "--model-kind",
            "ridge",
            "--split-mode",
            "random",
        ],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        [sys.executable, str(repo / "scripts/evaluate_submission.py"), "--truth", str(run / "labels_test.csv"), "--pred", str(run / "predictions.csv"), "--out", str(run / "eval_metrics.json")],
        cwd=repo,
        check=True,
    )
    assert (run / "model.joblib").exists()
    assert (run / "feature_columns.json").exists()
    assert (run / "predictions_val.csv").exists()
    assert (run / "predictions_test.csv").exists()
    metrics = json.loads((run / "eval_metrics.json").read_text())
    assert metrics["n"] > 0
    assert metrics["rmse"] >= 0


def test_train_baseline_accepts_protocol_split_mode(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    demo = tmp_path / "demo"
    run = tmp_path / "run_protocol"
    subprocess.run(
        [
            sys.executable,
            str(repo / "scripts/generate_demo_data.py"),
            "--out",
            str(demo),
            "--n-cells",
            "36",
            "--n-cycles",
            "50",
            "--seed",
            "11",
        ],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(repo / "scripts/train_baseline.py"),
            "--processed-dir",
            str(demo),
            "--out",
            str(run),
            "--max-cycle",
            "50",
            "--seed",
            "11",
            "--model-kind",
            "paper_ridge_loglife",
            "--split-mode",
            "protocol",
        ],
        cwd=repo,
        check=True,
    )
    assert (run / "log_metrics.json").exists()
    model_card = json.loads((run / "model_card.json").read_text())
    assert model_card["split_mode"] == "protocol"
