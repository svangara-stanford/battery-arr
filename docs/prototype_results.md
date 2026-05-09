# Prototype Results

This document records what the Battery-AAR phase-1 prototype demonstrates for NERSC AI-for-science reviewers. It is evidence of a runnable benchmark scaffold, not a claim of scientific model performance.

## What This Prototype Demonstrates

- The Attia/Chueh fast-charging protocol space can be generated locally.
- Public MatR JSON ZIP exports can be normalized into cell metadata and per-cycle summary tables.
- Early-cycle interpretable features can be built from the first 100 cycles.
- Scikit-learn baselines can be trained under deterministic seeds.
- Predictions can be evaluated through a hidden-evaluator-style `cell_id,y_pred` submission path.
- Leakage-aware split modes are wired in and failed split modes are reported instead of hidden.
- Local NERSC templates exist for preprocessing, baseline sweeps, and a non-LLM agent smoke test.

## Synthetic Demo Reproduction

```bash
python -m pip install -e ".[dev]"
pytest
make demo
```

Generated files include:

```text
data/demo/cell_metadata.csv
data/demo/cycle_summary.csv
data/demo/splits.csv
runs/demo_baseline/model.joblib
runs/demo_baseline/feature_columns.json
runs/demo_baseline/predictions.csv
runs/demo_baseline/metrics.json
runs/demo_baseline/eval_metrics.json
```

## Real-Data Baseline Matrix Reproduction

Manually download the public MatR raw data into:

```text
data/raw/chueh_toyota_fast_charge/
```

Raw data are not committed. The default matrix command is capped to four cells per batch so reviewers can run a small proof-of-concept path before scaling:

```bash
python scripts/run_real_data_baseline_matrix.py \
  --raw-dir data/raw/chueh_toyota_fast_charge \
  --processed-dir data/processed/chueh_toyota_fast_charge \
  --runs-dir runs/chueh_toyota_phase1 \
  --reports-dir reports \
  --max-cells-per-batch 4 \
  --max-cycle 100 \
  --seed 42
```

Use `--max-cells-per-batch 0` to attempt all cells after validating memory and runtime.

Generated files include:

```text
data/processed/chueh_toyota_fast_charge/cell_metadata.csv
data/processed/chueh_toyota_fast_charge/cycle_summary.csv
data/processed/chueh_toyota_fast_charge/qc_summary.json
data/processed/chueh_toyota_fast_charge/splits_<split_mode>.csv
runs/chueh_toyota_phase1/<split>_<model>/
reports/real_data_baseline_matrix.csv
reports/real_data_baseline_matrix.md
```

## Current Matrix Table

Generated locally from the MatR JSON ZIP files with `--max-cells-per-batch 4`, `--max-cycle 100`, and `--seed 42`. This is a capped smoke run. The batch and leave-one-batch-out modes fail explicitly because the capped labeled subset contains labeled cells from only one batch.

| split_mode | model_kind | status | n_train | n_val | n_test | rmse | mae | r2 | output_dir | failure_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random | dummy_mean | ok | 2.0 | 1.0 | 1.0 | 32.5000 | 32.5000 |  | runs/chueh_toyota_phase1/random_dummy_mean |  |
| random | ridge | ok | 2.0 | 1.0 | 1.0 | 42.9421 | 42.9421 |  | runs/chueh_toyota_phase1/random_ridge |  |
| random | random_forest | ok | 2.0 | 1.0 | 1.0 | 30.9250 | 30.9250 |  | runs/chueh_toyota_phase1/random_random_forest |  |
| random | gradient_boosting | ok | 2.0 | 1.0 | 1.0 | 3.4104 | 3.4104 |  | runs/chueh_toyota_phase1/random_gradient_boosting |  |
| batch | dummy_mean | failed |  |  |  |  |  |  | runs/chueh_toyota_phase1/batch_dummy_mean | batch split requires at least 3 labeled batches; found 1 |
| batch | ridge | failed |  |  |  |  |  |  | runs/chueh_toyota_phase1/batch_ridge | batch split requires at least 3 labeled batches; found 1 |
| batch | random_forest | failed |  |  |  |  |  |  | runs/chueh_toyota_phase1/batch_random_forest | batch split requires at least 3 labeled batches; found 1 |
| batch | gradient_boosting | failed |  |  |  |  |  |  | runs/chueh_toyota_phase1/batch_gradient_boosting | batch split requires at least 3 labeled batches; found 1 |
| protocol | dummy_mean | ok | 3.0 | 0.0 | 1.0 | 7.6667 | 7.6667 |  | runs/chueh_toyota_phase1/protocol_dummy_mean |  |
| protocol | ridge | ok | 3.0 | 0.0 | 1.0 | 11.7655 | 11.7655 |  | runs/chueh_toyota_phase1/protocol_ridge |  |
| protocol | random_forest | ok | 3.0 | 0.0 | 1.0 | 1.1600 | 1.1600 |  | runs/chueh_toyota_phase1/protocol_random_forest |  |
| protocol | gradient_boosting | ok | 3.0 | 0.0 | 1.0 | 11.8088 | 11.8088 |  | runs/chueh_toyota_phase1/protocol_gradient_boosting |  |
| leave_one_batch_out | dummy_mean | failed |  |  |  |  |  |  | runs/chueh_toyota_phase1/leave_one_batch_out_dummy_mean | leave_one_batch_out requires at least 2 labeled batches; found 1 |
| leave_one_batch_out | ridge | failed |  |  |  |  |  |  | runs/chueh_toyota_phase1/leave_one_batch_out_ridge | leave_one_batch_out requires at least 2 labeled batches; found 1 |
| leave_one_batch_out | random_forest | failed |  |  |  |  |  |  | runs/chueh_toyota_phase1/leave_one_batch_out_random_forest | leave_one_batch_out requires at least 2 labeled batches; found 1 |
| leave_one_batch_out | gradient_boosting | failed |  |  |  |  |  |  | runs/chueh_toyota_phase1/leave_one_batch_out_gradient_boosting | leave_one_batch_out requires at least 2 labeled batches; found 1 |

## Leakage Controls

- Training uses only early cycles, defaulting to `max_cycle=100`.
- Split modes include random smoke tests, batch groups, protocol groups, and leave-one-batch-out.
- Metadata IDs, protocol strings, batch IDs, and split labels are excluded from baseline feature columns.
- The evaluator submission file is restricted to `cell_id,y_pred`.
- Split modes that cannot produce the requested heldout structure are written as explicit failures in the matrix summary.

## Known Limitations

- The MatR loader has been smoke-tested on JSON ZIP exports, but full-batch loader validation is still needed before scientific claims.
- Current labels for JSON exports are explicit labels when present or conservative 80% capacity-threshold derivations when visible.
- The default real-data matrix is intentionally small and may not include enough labeled batches for every split mode.
- Baselines are simple tabular scikit-learn models, not state-of-the-art lifetime predictors.
- The current hidden-evaluator path is local only; a real benchmark needs filesystem isolation and locked labels.
- NERSC scripts are templates and are not optimized for Perlmutter I/O or queue policies.

## NERSC Proposal Readiness Claim

The prototype supports the proposal readiness claim by showing that the team can run a deterministic end-to-end battery lifetime prediction workflow: raw public data normalization, QC reports, leakage-aware split construction, baseline training, evaluator-style scoring, run summaries, and NERSC submission templates. The remaining work is scaling, stricter hidden evaluation, and validated transfer-split science.
