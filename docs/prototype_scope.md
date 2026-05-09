# Prototype scope

This is a proposal-strength proof of concept, not the final Battery-AAR benchmark.

## Included now

- modern Python package layout
- tests
- synthetic data path that runs locally
- deterministic splits
- baseline lifetime model
- hidden-label style evaluator
- first-pass MatR/HDF5 loader
- clear docs for raw data placement

## Deferred to Codex/Cursor follow-up

- exact hardening against all public MatR batch structures
- voltage-capacity curve interpolation and dQ/dV / dV/dQ features
- protocol-aware and batch-aware heldout splits
- model cards and data cards
- MLflow or W&B integration
- Slurm job scripts for NERSC
- neural sequence encoders
- agent controller and sandboxing
- true hidden server-side evaluator

## Prototype acceptance criterion

On a fresh MacBook environment:

```bash
python -m pip install -e ".[dev]"
pytest
make demo
```

should complete successfully and produce a trained baseline model plus metrics.
