from __future__ import annotations

from battery_aar.protocols.simulator import simulate_cycle_summary


def test_simulate_cycle_summary_is_stable_for_seed() -> None:
    protocol = {"cc1": 5.6, "cc2": 6.0, "cc3": 4.8, "cc4": 3.574}
    a = simulate_cycle_summary("cell-a", "batch-a", protocol, 900, n_cycles=5, seed=42)
    b = simulate_cycle_summary("cell-a", "batch-a", protocol, 900, n_cycles=5, seed=42)
    assert a.equals(b)
