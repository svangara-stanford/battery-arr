"""Protocol-space utilities for Attia/Chueh-style fast charging.

The public closed-loop optimization code uses a four-step parameterization where
CC1, CC2, and CC3 are free, while CC4 is constrained so that charging from
0% to 80% state of charge takes ten minutes with four 20%-SOC windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_C1 = (3.6, 4.0, 4.4, 4.8, 5.2, 5.6, 6.0, 7.0, 8.0)
DEFAULT_C2 = (3.6, 4.0, 4.4, 4.8, 5.2, 5.6, 6.0, 7.0)
DEFAULT_C3 = (3.6, 4.0, 4.4, 4.8, 5.2, 5.6)
DEFAULT_C4_LIMITS = (0.1, 4.81)


@dataclass(frozen=True)
class ChargingProtocol:
    """A four-step constant-current fast-charging protocol."""

    cc1: float
    cc2: float
    cc3: float
    cc4: float

    @property
    def readable(self) -> str:
        return f"{self.cc1:.1f}C-{self.cc2:.1f}C-{self.cc3:.1f}C-{self.cc4:.3f}C"

    def as_dict(self) -> dict[str, float | str]:
        return {
            "cc1": self.cc1,
            "cc2": self.cc2,
            "cc3": self.cc3,
            "cc4": self.cc4,
            "protocol_readable": self.readable,
        }


def compute_cc4(
    cc1: float,
    cc2: float,
    cc3: float,
    *,
    total_charge_minutes: float = 10.0,
    soc_fraction_per_step: float = 0.2,
) -> float:
    """Compute CC4 from CC1-CC3 under the 10-minute/0-80% SOC constraint.

    C-rates have units of inverse hours. Four constant-current windows each fill
    20% SOC, so `sum(0.2 / CC_i) = 10 / 60` hours.
    """

    total_hours = total_charge_minutes / 60.0
    remaining_hours = total_hours - soc_fraction_per_step * (
        1.0 / cc1 + 1.0 / cc2 + 1.0 / cc3
    )
    if remaining_hours <= 0:
        return float("nan")
    return soc_fraction_per_step / remaining_hours


def protocol_stress_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute simple interpretable stress features from protocol C-rates."""

    out = pd.DataFrame(index=df.index)
    currents = df[["cc1", "cc2", "cc3", "cc4"]].apply(pd.to_numeric, errors="coerce")
    out["protocol_mean_c_rate"] = currents.mean(axis=1)
    out["protocol_max_c_rate"] = currents.max(axis=1)
    out["protocol_rms_c_rate"] = np.sqrt(np.nanmean(currents.to_numpy(dtype=float) ** 2, axis=1))
    out["protocol_step_abs_mean"] = np.nanmean(np.abs(np.diff(currents.to_numpy(dtype=float), axis=1)), axis=1)
    out["protocol_cc4_minus_mean_first3"] = currents["cc4"] - currents[["cc1", "cc2", "cc3"]].mean(axis=1)
    return out


def generate_protocol_space(
    c1_values: tuple[float, ...] = DEFAULT_C1,
    c2_values: tuple[float, ...] = DEFAULT_C2,
    c3_values: tuple[float, ...] = DEFAULT_C3,
    c4_limits: tuple[float, float] = DEFAULT_C4_LIMITS,
    exclude_baseline: bool = True,
) -> pd.DataFrame:
    """Generate the 224 valid fast-charging protocols used in the CLO setup."""

    rows: list[dict[str, float | int | str]] = []
    for cc1, cc2, cc3 in product(c1_values, c2_values, c3_values):
        cc4 = compute_cc4(cc1, cc2, cc3)
        if not np.isfinite(cc4):
            continue
        if not (c4_limits[0] <= cc4 <= c4_limits[1]):
            continue
        if exclude_baseline and cc1 == 4.8 and cc2 == 4.8 and cc3 == 4.8:
            continue
        protocol = ChargingProtocol(cc1=cc1, cc2=cc2, cc3=cc3, cc4=cc4)
        rows.append(protocol.as_dict())

    df = pd.DataFrame(rows)
    df.insert(0, "protocol_id", np.arange(len(df), dtype=int))
    return df.round({"cc1": 3, "cc2": 3, "cc3": 3, "cc4": 3})


def save_protocol_space(path: str | Path) -> pd.DataFrame:
    """Generate and save the protocol space to CSV."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_protocol_space()
    df.to_csv(path, index=False)
    return df
