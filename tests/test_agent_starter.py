from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_candidate_runner_on_demo_data(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    demo = tmp_path / "demo"
    out = tmp_path / "candidate"
    subprocess.run(
        [
            sys.executable,
            str(repo / "scripts/generate_demo_data.py"),
            "--out",
            str(demo),
            "--n-cells",
            "24",
            "--n-cycles",
            "30",
            "--seed",
            "9",
        ],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(repo / "agent_starter/run_candidate.py"),
            "--processed-dir",
            str(demo),
            "--config",
            str(repo / "agent_starter/candidate_configs/ridge_capacity_slope.yaml"),
            "--out",
            str(out),
        ],
        cwd=repo,
        check=True,
    )
    result = json.loads((out / "candidate_result.json").read_text())
    assert result["model_kind"] == "ridge"
    assert result["n_test"] > 0
    assert (out / "predictions.csv").exists()
