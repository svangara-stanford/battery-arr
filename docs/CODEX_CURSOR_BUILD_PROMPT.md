# Detailed Codex/Cursor planning prompt

Use this prompt after opening the repository in Cursor or Codex.

---

You are a senior machine-learning research engineer helping build a credible Battery-AAR prototype for a NERSC AI-for-science proposal. The goal is not to overbuild the final benchmark yet; the goal is to turn this skeleton into a locally runnable, scientifically credible proof of concept for early-cycle battery lifetime prediction on the public Chueh/Toyota/Attia fast-charging dataset.

## Context

Project: Battery-AAR — outcome-gradable automated AI researchers for battery lifetime prediction.

Dataset: “Closed-loop optimization of extreme fast charging for batteries using machine learning” hosted at:

https://data.matr.io/1/projects/5d80e633f405260001c0b60a

Scientific paper to cite: Attia, Grover, Jin, et al., “Closed-loop optimization of fast-charging protocols for batteries with machine learning,” Nature 578, 397–402 (2020), DOI 10.1038/s41586-020-1994-5.

The public dataset consists of A123 APR18650M1A LFP/graphite cells charged under fast-charging protocols, with five batches of roughly 48 cells. The first four batches are CLO batches with about 100–120 cycles; the final validation batch is cycled to failure. Early prediction should use only early cycles, initially the first 100 cycles, and should avoid leakage through batch, protocol ID, file ordering, timestamps, or test labels.

The user also uploaded the public `chueh-ermon/battery-fast-charging-optimization` code. Treat it as reference code, not as modern package code. Useful files include `policies.py`, `sim_with_seed.py`, and `closed_loop_oed.py`.

## Current skeleton state

The repo already contains:

- `src/battery_aar/protocols/policy_space.py`: generates the 224 valid protocols.
- `src/battery_aar/protocols/simulator.py`: demo-only synthetic simulator.
- `src/battery_aar/data/matr_io.py`: best-effort loader for HDF5-backed MATLAB structs and CSVs.
- `src/battery_aar/features/early_cycle.py`: simple interpretable feature builder.
- `src/battery_aar/models/baseline.py`: scikit-learn baseline trainer.
- `src/battery_aar/evaluation/*`: metrics and evaluator helpers.
- `scripts/generate_demo_data.py`: creates a synthetic runnable dataset.
- `scripts/preprocess_chueh_toyota.py`: converts raw public data to normalized tables.
- `scripts/train_baseline.py`: trains and evaluates a baseline.
- tests for core modules and the end-to-end demo path.

## Phase 1 objective: make the proof of concept robust

Do the following in order. After each step, run `pytest` and keep the repo passing.

### 1. Validate the local demo path

Run:

```bash
python -m pip install -e ".[dev]"
pytest
make demo
```

Confirm that the following files are created:

- `data/demo/cell_metadata.csv`
- `data/demo/cycle_summary.csv`
- `runs/demo_baseline/metrics.json`
- `runs/demo_baseline/predictions.csv`
- `runs/demo_baseline/eval_metrics.json`

Inspect metrics and make sure `rmse`, `mae`, and `r2` are finite.

### 2. Download and inspect one real batch

Do not commit raw data. Manually download one small/raw batch file from MatR if possible and place it under:

```text
data/raw/chueh_toyota_fast_charge/batch_01/
```

Then inspect the file structure:

```python
import h5py
from pathlib import Path
p = next(Path("data/raw/chueh_toyota_fast_charge").rglob("*.mat"))
with h5py.File(p, "r") as f:
    print(list(f.keys()))
    def walk(name, obj):
        if len(name.split('/')) < 4:
            print(name, type(obj), getattr(obj, 'shape', None), getattr(obj, 'dtype', None))
    f.visititems(walk)
```

Update `src/battery_aar/data/matr_io.py` so it correctly extracts:

- `cell_id`
- `batch_id`
- `cycle_life`
- `protocol_readable` or equivalent
- `cc1`, `cc2`, `cc3`, `cc4` if present or derivable
- per-cycle `cycle_index`
- `discharge_capacity`
- `charge_capacity`
- `internal_resistance`
- `charge_time`
- temperature fields if available

Add a unit test with a tiny synthetic HDF5 fixture that matches the discovered field layout.

### 3. Harden processed data schema and quality checks

Add validation functions that check:

- no duplicate `cell_id` in metadata
- `cycle_life` is present for training rows
- `cycle_index` starts at 1 or is consistently zero-indexed and normalized
- discharge capacity is positive where present
- no cell in `cycle_summary` is missing from metadata
- no overlap between train/val/test splits

Write a report to `reports/qc_summary.json` during preprocessing with counts by batch, missingness by column, cycles per cell, and basic cycle-life statistics.

### 4. Add stronger early-cycle features

Extend `features/early_cycle.py` with scientifically meaningful but still simple features:

- capacity at cycles 2, 10, 50, and 100 when available
- log absolute capacity fade between early and late early cycles
- linear slope of discharge capacity over cycles 2–100
- curvature proxy using two slopes: cycles 2–50 and 50–100
- internal resistance mean/slope if available
- charge time mean/slope
- temperature mean/max if available
- protocol C-rates and simple protocol stress features

Keep features interpretable and documented. Add tests for missing cycles and missing optional columns.

### 5. Improve split strategies

Add split modes:

- random cell split for smoke tests
- group split by `batch_id`
- group split by `protocol_readable` or protocol cluster
- leave-one-batch-out split

The default should remain local and simple, but the docs must explain that final scientific claims require group/transfer splits.

### 6. Add baseline model comparison

In `models/baseline.py`, support:

- `dummy_mean`
- `ridge`
- `random_forest`
- optional `gradient_boosting`

Update `scripts/train_baseline.py` to accept `--model-kind` and save:

- model artifact
- feature column list
- predictions for val/test
- metrics by split
- a short JSON model card

### 7. Add hidden evaluator behavior

Make `scripts/evaluate_submission.py` accept a predictions CSV with only `cell_id,y_pred`. The truth CSV should not be used by training scripts except for the local demo. Add a `docs/evaluator_design.md` explaining how this becomes a real hidden evaluator later: locked labels not mounted in agent sandboxes, submission budgets, final holdout, leakage checks, and seed replication.

### 8. Add README evidence for NERSC reviewers

Update README so the top section clearly says:

- This is a locally runnable prototype.
- It demonstrates protocol generation, preprocessing path, baseline model training, and evaluator path.
- It does not require API keys.
- It requires manual download of the public data into `data/raw/chueh_toyota_fast_charge/`.
- It can run without real data using `make demo`.

### 9. Keep quality high

Engineering standards:

- Type hints on public functions.
- Short pure functions where possible.
- No hidden network calls.
- No raw data committed.
- Deterministic seeds.
- `pytest` must pass.
- Do not silence loader failures without surfacing clear errors.
- Add docstrings where scientific assumptions are encoded.

## Phase 2 stretch goals

Only after Phase 1 is stable:

1. Add voltage-capacity interpolation arrays and dQ/dV features.
2. Add simple self-supervised sequence baseline using PyTorch.
3. Add Slurm script templates under `nersc/`.
4. Add MLflow local tracking.
5. Add an agent-sandbox directory with a dummy agent that proposes a config and submits predictions.

## Definition of done for this handoff

The repo is successful when a new reviewer can run:

```bash
python -m pip install -e ".[dev]"
pytest
make demo
```

and see a complete baseline training/evaluation path. With manually downloaded raw data, they should also be able to run:

```bash
python scripts/preprocess_chueh_toyota.py --raw-dir data/raw/chueh_toyota_fast_charge --out data/processed/chueh_toyota_fast_charge --max-cells-per-batch 4 --first-n-cycles 100
python scripts/train_baseline.py --processed-dir data/processed/chueh_toyota_fast_charge --out runs/chueh_toyota_baseline --max-cycle 100 --seed 42
```

Do not claim scientific performance until the loader is validated on the real public batches and transfer splits are implemented.
