# Chueh/Ermon public code notes

The user uploaded `battery-fast-charging-optimization-master.zip`, corresponding to the public repository:

https://github.com/chueh-ermon/battery-fast-charging-optimization

This prototype does not vendor the full legacy code. Instead, it reimplements a small, testable subset needed for the local proof of concept:

- protocol-space generation in `src/battery_aar/protocols/policy_space.py`
- a demo-only synthetic simulator in `src/battery_aar/protocols/simulator.py`
- modern package, scripts, and tests around baseline lifetime prediction

The uploaded legacy repository is useful reference material for future development, especially `policies.py`, `sim_with_seed.py`, and `closed_loop_oed.py`.
