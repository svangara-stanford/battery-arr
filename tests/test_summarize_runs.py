from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_summarize_runs_on_tiny_fake_run(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    run = tmp_path / "runs/demo"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(
        json.dumps({"test": {"rmse": 1.0, "mae": 0.5, "r2": 0.1, "n": 2}})
    )
    (run / "model_card.json").write_text(
        json.dumps({"model_kind": "dummy_mean", "max_cycle": 5, "seed": 42})
    )
    pd.DataFrame({"cell_id": ["a", "b"], "y_pred": [1.0, 2.0]}).to_csv(
        run / "predictions.csv",
        index=False,
    )
    subprocess.run(
        [
            sys.executable,
            str(repo / "scripts/summarize_runs.py"),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--reports-dir",
            str(tmp_path / "reports"),
        ],
        cwd=repo,
        check=True,
    )
    summary = pd.read_csv(tmp_path / "reports/run_summary.csv")
    assert summary.loc[0, "run_name"] == "demo"
    assert summary.loc[0, "rmse"] == 1.0
    assert (tmp_path / "reports/run_summary.md").exists()
