# Uploaded legacy code audit

Uploaded archive: `battery-fast-charging-optimization-master.zip`.

Corresponding public repository: `chueh-ermon/battery-fast-charging-optimization`.

Useful files observed in the upload:

- `policies.py`: generates the four-step fast-charging protocol space. The prototype reimplements this logic in `src/battery_aar/protocols/policy_space.py` and tests that 224 valid protocols are produced.
- `sim_with_seed.py`: stochastic thermal/degradation-inspired simulator for protocol lifetime. The prototype does not copy this simulator; it uses a clearly marked demo-only synthetic simulator for smoke tests.
- `closed_loop_oed.py`: closed-loop optimal experimental design script. This is useful future reference for active-selection experiments but is not required for the local baseline proof of concept.
- `data/policies_all.csv`, `data/batch`, `data/pred`, and `data/bounds`: legacy data artifacts used by the CLO scripts. They are not committed into this modern skeleton.

Reason for not vendoring the legacy code:

- The uploaded code is a compact research-code repository from 2020, implemented for Python 3.7 and paper-specific scripts.
- The NERSC prototype needs a standard ML repo layout with tests, reproducible local commands, and clear data boundaries.
- The skeleton therefore reimplements only the minimal tested logic needed for proof-of-concept operation.
