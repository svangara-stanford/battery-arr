from __future__ import annotations

import numpy as np

from battery_aar.protocols.policy_space import compute_cc4, generate_protocol_space


def test_generate_protocol_space_has_expected_224_protocols() -> None:
    df = generate_protocol_space()
    assert len(df) == 224
    assert {"cc1", "cc2", "cc3", "cc4", "protocol_readable"}.issubset(df.columns)
    assert df["cc4"].between(0.1, 4.81).all()
    assert not ((df["cc1"] == 4.8) & (df["cc2"] == 4.8) & (df["cc3"] == 4.8)).any()


def test_compute_cc4_satisfies_ten_minute_constraint() -> None:
    cc4 = compute_cc4(3.6, 6.0, 5.6)
    total_hours = 0.2 / 3.6 + 0.2 / 6.0 + 0.2 / 5.6 + 0.2 / cc4
    assert np.isclose(total_hours, 10 / 60)
