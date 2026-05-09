# NERSC Scaffold

These files are templates for moving the local Battery-AAR prototype onto NERSC. They are intentionally simple and should be adapted to the allocation, queue, and filesystem policy used by a real project.

## Expected Workflow

Use either a Python environment on the login node for lightweight setup or a container image built from `containers/Containerfile` for more reproducible batch runs. On Perlmutter, a typical flow is:

```bash
module load python
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

For a container workflow, build the image with the site-supported container tooling, then launch commands through the center-supported runtime. The provided Slurm templates leave image paths as placeholders.

## Suggested Directory Layout

```text
$SCRATCH/battery-aar/
  repo/                         # git checkout
  data/raw/chueh_toyota_fast_charge/
  data/processed/chueh_toyota_fast_charge/
  runs/
  reports/
```

Raw MatR ZIP files should live under `data/raw/chueh_toyota_fast_charge/`. They are manually downloaded public data and should not be committed to git.

## Preprocessing

```bash
python scripts/preprocess_chueh_toyota.py \
  --raw-dir data/raw/chueh_toyota_fast_charge \
  --out data/processed/chueh_toyota_fast_charge \
  --max-cells-per-batch 4 \
  --first-n-cycles 100
```

The `--max-cells-per-batch 4` setting is a smoke-test cap. Remove or increase it for larger real-data runs after validating runtime and memory.

## Baseline Sweeps

```bash
python scripts/run_real_data_baseline_matrix.py \
  --raw-dir data/raw/chueh_toyota_fast_charge \
  --processed-dir data/processed/chueh_toyota_fast_charge \
  --runs-dir runs/chueh_toyota_phase1 \
  --reports-dir reports
```

The sweep writes per-run artifacts under `runs/chueh_toyota_phase1/` and summary tables under `reports/`.

## Slurm Templates

- `submit_preprocess.slurm`: preprocesses public raw data into normalized tables.
- `submit_baseline_sweep.slurm`: runs the phase-1 baseline matrix.
- `submit_agent_smoke.slurm`: runs a non-LLM candidate config through the local evaluator path.

Set partition, time, image paths, and project-specific environment details before submission. The templates intentionally do not hard-code account names.

## Current Limitations

- The prototype has not been optimized for Perlmutter I/O, parallel JSON parsing, or large full-batch preprocessing.
- The current agent starter does not call LLM APIs.
- Local demo labels are visible to the runner; hidden-label security requires a separate evaluator job and filesystem isolation.
- Scientific claims require validated real-data loading and leakage-aware transfer splits with enough labeled cells.
