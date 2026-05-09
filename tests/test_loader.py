from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from battery_aar.data.matr_io import load_hdf5_mat_batch


def _write_string_dataset(h: h5py.File, name: str, text: str) -> h5py.Reference:
    ds = h.create_dataset(name, data=np.array([ord(c) for c in text], dtype="uint16"))
    return ds.ref


def test_load_hdf5_mat_batch_synthetic_fixture(tmp_path: Path) -> None:
    path = tmp_path / "batch_fixture.mat"
    with h5py.File(path, "w") as h:
        batch = h.create_group("batch")
        batch.create_dataset("cycle_life", data=np.array([[1000.0, 900.0]]))
        policy_refs = np.empty((1, 2), dtype=h5py.ref_dtype)
        summary_refs = np.empty((1, 2), dtype=h5py.ref_dtype)
        for idx, (text, q0) in enumerate([("3.6C-6.0C-5.6C-4.0C", 1.1), ("4.0C-5.6C-5.2C-4.1C", 1.08)]):
            policy_refs[0, idx] = _write_string_dataset(h, f"policy_{idx}", text)
            g = h.create_group(f"summary_{idx}")
            g.create_dataset("cycle", data=np.arange(1, 4))
            g.create_dataset("QDischarge", data=np.array([q0, q0 - 0.01, q0 - 0.02]))
            g.create_dataset("IR", data=np.array([0.017, 0.018, 0.019]))
            summary_refs[0, idx] = g.ref
        batch.create_dataset("policy", data=policy_refs)
        batch.create_dataset("summary", data=summary_refs)

    loaded = load_hdf5_mat_batch(path, first_n_cycles=2)
    assert len(loaded.metadata) == 2
    assert loaded.metadata.loc[0, "cycle_life"] == 1000.0
    assert loaded.metadata.loc[0, "cc1"] == 3.6
    assert loaded.cycle_summary["cycle_index"].max() == 2
    assert loaded.cycle_summary["cell_id"].nunique() == 2
