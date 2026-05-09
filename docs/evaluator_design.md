# Evaluator design

The local evaluator is intentionally simple: it accepts a hidden-truth CSV and a submission CSV with exactly `cell_id,y_pred`, aligns rows by `cell_id`, and computes RMSE, MAE, R², and sample count. The prediction file must not include `cycle_life` or other label-like columns.

For `make demo`, `scripts/train_baseline.py` writes `labels_test.csv` only so the repository is runnable on one machine. That file is a local-demo convenience, not the benchmark design.

For the full Battery-AAR benchmark, this interface should evolve into a hidden evaluator with the following constraints:

1. Locked test labels are never mounted into agent sandboxes.
2. Agents can see training labels and validation feedback only through controlled submissions.
3. Each run has a submission budget to discourage validation-set hill-climbing.
4. Final results are scored on a heldout test split that is not used for iteration.
5. Transfer splits should include heldout cells, heldout protocols, heldout batches, and eventually leave-one-dataset-out evaluation.
6. Leakage checks should flag features derived from file ordering, timestamps, cycler IDs, batch IDs used as ordinary predictive features, protocol IDs that encode holdout membership, or labels accidentally copied into feature tables.
7. Final submissions should be replicated across deterministic seeds and checked for physically plausible feature use.
8. The final holdout should be mounted only inside the evaluator job, never inside the agent training sandbox.
9. The evaluator should log immutable submission artifacts: code revision, config, random seeds, prediction checksum, and scorer version.

The local script `scripts/evaluate_submission.py` is the minimal executable version of this path. It is not secure by itself; it is only a proof-of-concept interface.
