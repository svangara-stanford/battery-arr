# Scientific Baselines

This document separates infrastructure baselines from scientifically motivated baselines. None of the current results should be described as an exact reproduction of the Attia/Chueh or Severson early-prediction papers until the original feature matrices, trained feature indices, scaling constants, and transfer splits are fully present.

## Generic Sklearn Baseline

The current generic baseline uses `battery_aar.features.early_cycle.build_early_cycle_features` and ordinary scikit-learn regressors:

- `dummy_mean`
- `ridge`
- `random_forest`
- `gradient_boosting`

It is useful for validating the local benchmark infrastructure: preprocessing, split assignment, feature construction, model artifacts, prediction CSVs, and evaluator behavior. It should not be interpreted as the paper early-prediction model.

## Paper-Inspired Early-Prediction Baseline

The new `paper_ridge_loglife` model uses `battery_aar.features.paper_features.build_paper_features`. It is inspired by the BMS-autoanalysis early-cycle feature design used around the Severson/Attia battery lifetime work:

- It trains a ridge model on `log10(cycle_life)`.
- It predicts in log space, then back-transforms with `10 ** y_pred_log10`.
- It saves cycle-life metrics in `metrics.json`.
- It saves log-space metrics in `log_metrics.json`.

Implemented direct or close scalar analogs:

| Feature | Status |
| --- | --- |
| QDischarge cycle 2 | Direct scalar analog from `discharge_capacity` |
| QDischarge cycle 100 | Direct scalar analog from `discharge_capacity` |
| Max capacity change through cycle 100 | Scalar range over cycles 2-100; paper-inspired approximation |
| Linear fit of QDischarge over cycles 2-100 | Direct scalar analog |
| Linear fit of QDischarge over cycles 91-100 | Direct scalar analog |
| QDischarge cycle 100 minus cycle 10 | Scalar approximation to early capacity difference |
| Log absolute capacity difference | Scalar approximation |
| Sum and squared-sum Q difference | Scalar approximations unless Qdlin traces are present |
| Energy difference between cycles 10 and 100 | Direct when `discharge_energy` is available |

Curve-derived BMS-autoanalysis features such as Qdlin statistics over voltage grids are only computed when per-cycle interpolated arrays are present in the processed table. The current MatR JSON ZIP preprocessing does not yet persist full interpolated Q(V)/V(Q) traces, so `paper_curve_*` features are present as missing values and `paper_curve_features_available` marks that state.

## Future BayesGap Closed-Loop Optimization Baseline

The uploaded Chueh/Attia fast-charging optimization code is most relevant for:

- protocol-space construction,
- simulator behavior,
- BayesGap closed-loop selection logic.

A future benchmark baseline should reproduce or wrap the BayesGap closed-loop optimizer against a fixed protocol space and a locked evaluator. That is separate from early-cycle lifetime prediction. It should be added only after the real-data loader and hidden split protocol are stable.

## Current Claim Boundary

The repository now contains:

- an infrastructure baseline,
- a paper-inspired log-life early-prediction baseline,
- NERSC run scaffolding,
- a local evaluator-style submission path.

It does not yet contain an exact BMS-autoanalysis reproduction or a validated BayesGap closed-loop benchmark.
