# Agent Starter

This directory demonstrates the Battery-AAR candidate/evaluator pattern without calling any LLM API.

The flow is:

1. A candidate YAML file proposes a model kind and a small interpretable feature subset.
2. `run_candidate.py` trains on the processed training split.
3. The runner writes a submission-style `predictions.csv` containing only `cell_id,y_pred`.
4. For local demos only, the runner evaluates against heldout labels present in the processed data and writes `candidate_result.json`.

Example:

```bash
python agent_starter/run_candidate.py \
  --processed-dir data/demo \
  --config agent_starter/candidate_configs/ridge_capacity_slope.yaml \
  --out runs/agent_smoke/ridge_capacity_slope
```

In a hidden evaluator, labels would not be mounted in the agent sandbox. The local evaluation step is only a proof-of-concept interface.
