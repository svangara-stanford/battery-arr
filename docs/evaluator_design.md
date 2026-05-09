# Evaluator design sketch

The local evaluator is intentionally simple: it accepts a truth CSV and a predictions CSV, aligns rows by `cell_id`, and computes RMSE, MAE, R², and sample count.

For the full Battery-AAR benchmark, this should evolve into a hidden evaluator with the following constraints:

1. Locked test labels are never mounted into agent sandboxes.
2. Agents can see training labels and validation feedback only through controlled submissions.
3. Each run has a submission budget to discourage validation-set hill-climbing.
4. Final results are scored on a heldout test split that is not used for iteration.
5. Transfer splits should include heldout cells, heldout protocols, heldout batches, and eventually leave-one-dataset-out evaluation.
6. Leakage checks should flag features derived from file ordering, timestamps, cycler IDs, batch IDs used inappropriately, or labels accidentally copied into feature tables.
7. Top methods should be replicated across seeds and ablated for physical plausibility.

The local script `scripts/evaluate_submission.py` is the minimal executable version of this path. It is not secure by itself; it is only a proof-of-concept interface.
