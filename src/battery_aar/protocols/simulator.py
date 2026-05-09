"""Demo-only synthetic lifetime and cycle-summary simulator.

This module exists so the repository is runnable before the large public MatR
batches are downloaded. It is not the original thermal/degradation simulator
and must not be used for scientific conclusions.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd


def protocol_stress(cc1: float, cc2: float, cc3: float, cc4: float) -> float:
    """Return a simple stress score for a protocol.

    Higher C-rates, sharp current steps, and high late-stage current all
    increase stress.
    """

    currents = np.asarray([cc1, cc2, cc3, cc4], dtype=float)
    rms = math.sqrt(float(np.mean(currents**2)))
    step_penalty = float(np.mean(np.abs(np.diff(currents))))
    late_penalty = max(cc4 - 4.2, 0.0)
    return rms + 0.35 * step_penalty + 0.8 * late_penalty


def simulate_cycle_life(
    protocol: Mapping[str, float],
    *,
    seed: int = 0,
    noise_std: float = 60.0,
) -> int:
    """Generate a reproducible synthetic lifetime for demos and tests."""

    cc1 = float(protocol["cc1"])
    cc2 = float(protocol["cc2"])
    cc3 = float(protocol["cc3"])
    cc4 = float(protocol["cc4"])
    rng_seed = int(seed + round(100 * cc1) + round(200 * cc2) + round(300 * cc3) + round(400 * cc4))
    rng = np.random.default_rng(rng_seed)

    stress = protocol_stress(cc1, cc2, cc3, cc4)
    smoothness_bonus = 45.0 / (1.0 + float(np.var([cc1, cc2, cc3, cc4])))
    base = 1550.0 - 115.0 * stress + smoothness_bonus
    lifetime = base + rng.normal(0.0, noise_std)
    return max(50, int(round(lifetime)))


def simulate_cycle_summary(
    cell_id: str,
    batch_id: str,
    protocol: Mapping[str, float],
    cycle_life: int,
    *,
    n_cycles: int = 100,
    seed: int = 0,
) -> pd.DataFrame:
    """Simulate early-cycle summary rows with degradation correlated to lifetime."""

    rng = np.random.default_rng(abs(hash((cell_id, seed))) % (2**32))
    cycles = np.arange(1, n_cycles + 1)
    stress = protocol_stress(
        float(protocol["cc1"]), float(protocol["cc2"]), float(protocol["cc3"]), float(protocol["cc4"])
    )
    initial_capacity = rng.normal(1.10, 0.015)
    # Lower lifetime -> stronger early fade. The scale is deliberately mild.
    fade_per_cycle = 0.000015 + 0.00010 * (1200.0 / max(float(cycle_life), 100.0)) + stress * 0.000006
    curvature = 2.5e-7 * (1200.0 / max(float(cycle_life), 100.0))
    noise = rng.normal(0.0, 0.0025, size=n_cycles)
    discharge_capacity = initial_capacity - fade_per_cycle * cycles - curvature * cycles**2 + noise
    charge_capacity = discharge_capacity + rng.normal(0.008, 0.002, size=n_cycles)
    internal_resistance = 0.017 + 0.000012 * cycles + 0.00015 * stress + rng.normal(0, 0.00025, n_cycles)
    charge_time = 10.0 + 0.0025 * cycles + 0.10 * max(stress - 5.0, 0.0) + rng.normal(0, 0.025, n_cycles)
    temperature_mean = 30.0 + 0.35 * stress + rng.normal(0, 0.35, n_cycles)
    temperature_max = temperature_mean + rng.normal(1.0, 0.15, n_cycles)
    temperature_min = temperature_mean - rng.normal(0.8, 0.15, n_cycles)

    return pd.DataFrame(
        {
            "cell_id": cell_id,
            "batch_id": batch_id,
            "cycle_index": cycles,
            "discharge_capacity": np.maximum(discharge_capacity, 0.1),
            "charge_capacity": np.maximum(charge_capacity, 0.1),
            "internal_resistance": internal_resistance,
            "temperature_max": temperature_max,
            "temperature_mean": temperature_mean,
            "temperature_min": temperature_min,
            "charge_time": charge_time,
        }
    )
