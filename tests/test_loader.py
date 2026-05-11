from __future__ import annotations

import json
import zipfile
from pathlib import Path

import h5py
import numpy as np

from battery_aar.data.matr_io import load_hdf5_mat_batch, load_matr_json_zip, load_raw_batches


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
            g.create_dataset("cycle", data=np.arange(0, 3))
            g.create_dataset("QDischarge", data=np.array([q0, q0 - 0.01, q0 - 0.02]))
            g.create_dataset("dc_internal_resistance", data=np.array([0.017, 0.018, 0.019]))
            g.create_dataset("temperature_average", data=np.array([30.0, 31.0, 32.0]))
            g.create_dataset("charge_duration", data=np.array([600.0, 599.0, 598.0]))
            summary_refs[0, idx] = g.ref
        batch.create_dataset("policy", data=policy_refs)
        batch.create_dataset("summary", data=summary_refs)

    loaded = load_hdf5_mat_batch(path, first_n_cycles=2)
    assert len(loaded.metadata) == 2
    assert loaded.metadata.loc[0, "cycle_life"] == 1000.0
    assert loaded.metadata.loc[0, "cc1"] == 3.6
    assert loaded.cycle_summary["cycle_index"].max() == 2
    assert loaded.cycle_summary["cycle_index"].min() == 1
    assert "internal_resistance" in loaded.cycle_summary
    assert loaded.cycle_summary["charge_time"].notna().any()
    assert loaded.cycle_summary["cell_id"].nunique() == 2


def test_load_matr_json_zip_beep_structure(tmp_path: Path) -> None:
    path = tmp_path / "2018-08-28_oed_0.zip"
    summary = {
        "cycle_index": list(range(30)),
        "discharge_capacity": [1.0] * 15 + [0.79] * 15,
        "charge_capacity": [1.02] * 30,
        "dc_internal_resistance": [0.017 + 0.0001 * i for i in range(30)],
        "temperature_maximum": [35.0] * 30,
        "temperature_average": [30.0] * 30,
        "temperature_minimum": [25.0] * 30,
        "charge_duration": [600.0] * 30,
    }
    obj = {
        "@module": "beep.structure.arbin",
        "@class": "ArbinDatapath",
        "metadata": {"barcode": "EL150800746390"},
        "protocol": r"OED\20180828-5pt6_6_4pt8_3pt574.sdu",
        "summary": summary,
    }
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("2018-08-28_oed_0_CH1_structure.json", json.dumps(obj))

    loaded = load_matr_json_zip(path, first_n_cycles=10)
    assert loaded.metadata.loc[0, "cell_id"] == "2018-08-28_oed_0_EL150800746390"
    assert loaded.metadata.loc[0, "cc1"] == 5.6
    assert loaded.metadata.loc[0, "cycle_life"] == 16
    assert loaded.cycle_summary["cycle_index"].min() == 1
    assert loaded.cycle_summary["cycle_index"].max() == 10
    assert loaded.cycle_summary.loc[0, "internal_resistance"] == 0.017


def test_load_matr_json_zip_skips_member_without_summary(tmp_path: Path) -> None:
    path = tmp_path / "2018-08-28_oed_0.zip"
    good = {
        "metadata": {"barcode": "good"},
        "protocol": r"OED\20180828-5pt6_6_4pt8_3pt574.sdu",
        "summary": {
            "cycle_index": [0, 1, 2],
            "discharge_capacity": [1.0, 0.99, 0.98],
        },
    }
    bad = {"metadata": {"barcode": "bad"}, "protocol": "missing_summary.sdu"}
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("2018-08-28_oed_0_CH1_structure.json", json.dumps(good))
        zf.writestr("2018-08-28_oed_0_CH2_structure.json", json.dumps(bad))

    loaded = load_matr_json_zip(path, first_n_cycles=3)
    assert len(loaded.metadata) == 1
    assert loaded.metadata.loc[0, "batch_id"] == "2018-08-28_oed_0"
    assert loaded.parse_errors
    assert "missing top-level summary" in loaded.parse_errors[0]


def test_load_matr_json_zip_handles_raw_data_sibling(tmp_path: Path) -> None:
    path = tmp_path / "2018-08-28_oed_0.zip"
    obj = {
        "raw_data": {"unused": [[float(i)] * 5 for i in range(50)]},
        "metadata": {"barcode": "raw_sibling"},
        "protocol": r"OED\20180828-5pt6_6_4pt8_3pt574.sdu",
        "summary": {
            "cycle_index": list(range(30)),
            "discharge_capacity": [1.0] * 15 + [0.79] * 15,
        },
    }
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("2018-08-28_oed_0_CH1_structure.json", json.dumps(obj))

    loaded = load_matr_json_zip(path, first_n_cycles=10)
    assert loaded.metadata.loc[0, "cell_id"] == "2018-08-28_oed_0_raw_sibling"
    assert loaded.metadata.loc[0, "cycle_life"] == 16
    assert loaded.cycle_summary["cycle_index"].max() == 10


def test_load_raw_batches_detects_multiple_zip_batch_ids_from_paths(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    summary = {
        "cycle_index": [0, 1, 2],
        "discharge_capacity": [1.0, 0.99, 0.98],
        "charge_capacity": [1.01, 1.0, 0.99],
    }
    for batch_id in ("2018-08-28_oed_0", "2019-01-24_batch9"):
        obj = {
            "metadata": {"barcode": f"{batch_id}_barcode"},
            "protocol": r"OED\20180828-5pt6_6_4pt8_3pt574.sdu",
            "summary": summary,
        }
        with zipfile.ZipFile(raw / f"{batch_id}.zip", "w") as zf:
            zf.writestr(f"{batch_id}_CH1_structure.json", json.dumps(obj))

    loaded = load_raw_batches(raw, first_n_cycles=3)
    assert set(loaded.metadata["batch_id"]) == {"2018-08-28_oed_0", "2019-01-24_batch9"}
    assert loaded.cycle_summary["batch_id"].nunique() == 2
