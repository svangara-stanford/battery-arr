from __future__ import annotations

import re
from pathlib import Path


def test_nersc_templates_exist_and_are_not_user_specific() -> None:
    repo = Path(__file__).resolve().parents[1]
    paths = [
        repo / "containers/Containerfile",
        repo / "nersc/README.md",
        repo / "nersc/submit_preprocess.slurm",
        repo / "nersc/submit_baseline_sweep.slurm",
        repo / "nersc/submit_agent_smoke.slurm",
    ]
    for path in paths:
        assert path.exists(), path
        text = path.read_text()
        assert "sreyavangara" not in text
        assert "/Users/" not in text
        assert not re.search(r"^#SBATCH\s+(--account|-A)\b", text, flags=re.MULTILINE)
