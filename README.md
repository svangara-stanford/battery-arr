# Battery-AAR Prototype

A lightweight, local-first machine-learning scaffold for the NERSC AI proposal **Battery-AAR: Outcome-Gradable Automated AI Researchers for Battery Lifetime Prediction**.

This repo is intentionally small enough to run on a MacBook, while preserving the shape of a serious ML research codebase that can later scale to NERSC: deterministic preprocessing, train/validation/test splits, feature extraction, baseline training, hidden-label evaluation, tests, and a data layout that avoids committing large raw files.

## What works now

- Generate the **224 valid four-step fast-charging protocols** from the Attia/Chueh closed-loop optimization setup.
- Create a small synthetic/demo dataset so the repository is runnable before the public MatR batches are downloaded.
- Load best-effort MATLAB/HDF5 MatR batch structs and already-tabular CSVs into a normalized schema.
- Build interpretable early-cycle tabular features from cycle summaries.
- Train a reproducible scikit-learn baseline lifetime regressor.
- Score predictions through an evaluator that can hide test labels.
- Run unit and smoke tests for the protocol space, data splits, features, metrics, loader, and end-to-end demo.

## Scientific target

The intended real dataset is the TRI/Stanford/MIT/Toyota public project:

**Closed-loop optimization of extreme fast charging for batteries using machine learning**  
https://data.matr.io/1/projects/5d80e633f405260001c0b60a

The uploaded public code corresponds to the Chueh/Ermon/Attia closed-loop optimization repository:

https://github.com/chueh-ermon/battery-fast-charging-optimization

This prototype does **not** commit raw battery batches. Download the large batch files manually and place them under `data/raw/chueh_toyota_fast_charge/`.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
make demo
```

After `make demo`, inspect:

```text
data/demo/cell_metadata.csv
data/demo/cycle_summary.csv
runs/demo_baseline/metrics.json
runs/demo_baseline/model.joblib
runs/demo_baseline/predictions.csv
runs/demo_baseline/eval_metrics.json
```

## Real-data layout

Place downloaded public files here:

```text
data/raw/chueh_toyota_fast_charge/
  batch_01/
    *.mat   # MATLAB struct batch files, if downloaded
    *.csv   # raw or processed cell CSVs, if downloaded
  batch_02/
    ...
```

Preprocess a small subset first:

```bash
python scripts/preprocess_chueh_toyota.py \
  --raw-dir data/raw/chueh_toyota_fast_charge \
  --out data/processed/chueh_toyota_fast_charge \
  --max-cells-per-batch 4 \
  --first-n-cycles 100
```

Then train:

```bash
python scripts/train_baseline.py \
  --processed-dir data/processed/chueh_toyota_fast_charge \
  --out runs/chueh_toyota_baseline \
  --max-cycle 100 \
  --seed 42
```

## Repository structure

```text
src/battery_aar/
  data/          MatR/HDF5 loading, processed schema, deterministic splits
  features/      early-cycle and voltage-curve feature utilities
  models/        sklearn baseline models
  evaluation/    metrics and hidden-label evaluator helpers
  protocols/     fast-charge protocol generation and demo simulator
  utils/         seeds and lightweight I/O helpers
scripts/         runnable local workflows
configs/         local baseline configuration
tests/           unit and smoke tests
docs/            dataset notes, prototype scope, Codex/Cursor build prompt
external/        notes for third-party code; no large vendored code
```

## Data, API keys, and secrets

No API key is required for the local prototype.

You need to add:

1. **Raw data:** download the MatR batches from the public project page and place them under `data/raw/chueh_toyota_fast_charge/`.
2. **Optional experiment tracking:** add `WANDB_API_KEY` or `MLFLOW_TRACKING_URI` only if you decide to use those services.
3. **Optional agent infrastructure:** add `OPENAI_API_KEY` or `STANFORD_AI_PLAYGROUND_API_KEY` only when implementing the future agentic researcher layer. The current code does not call any LLM APIs.

Use `.env.example` as the template. Do not commit `.env`, raw data, processed data, checkpoints, or run outputs.

## Acceptance test for a light NERSC prototype

A reviewer should be able to run:

```bash
python -m pip install -e ".[dev]"
pytest
make demo
```

and see passing tests plus a baseline run with saved predictions and metrics. That is the intended “locally running prototype” proof point.
