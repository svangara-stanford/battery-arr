#!/usr/bin/env python
"""Generate a tiny synthetic dataset for local smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from battery_aar.data.schema import (
    qc_summary,
    validate_processed_tables,
    validate_split_assignments,
)
from battery_aar.data.split import SplitConfig, make_split_assignments
from battery_aar.protocols.policy_space import generate_protocol_space
from battery_aar.protocols.simulator import simulate_cycle_life, simulate_cycle_summary
from battery_aar.utils.io import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/demo"))
    parser.add_argument("--n-cells", type=int, default=72)
    parser.add_argument("--n-cycles", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = ensure_dir(args.out)
    protocols = generate_protocol_space().sample(args.n_cells, random_state=args.seed).reset_index(drop=True)

    metadata_rows = []
    cycle_parts = []
    for i, protocol in protocols.iterrows():
        batch_id = f"demo_batch_{i % 4 + 1:02d}"
        cell_id = f"demo_cell_{i:04d}"
        proto = protocol.to_dict()
        cycle_life = simulate_cycle_life(proto, seed=args.seed + i)
        metadata_rows.append(
            {
                "cell_id": cell_id,
                "batch_id": batch_id,
                "cycle_life": cycle_life,
                "protocol_readable": proto["protocol_readable"],
                "cc1": proto["cc1"],
                "cc2": proto["cc2"],
                "cc3": proto["cc3"],
                "cc4": proto["cc4"],
            }
        )
        cycle_parts.append(
            simulate_cycle_summary(
                cell_id,
                batch_id,
                proto,
                cycle_life,
                n_cycles=args.n_cycles,
                seed=args.seed + i,
            )
        )

    metadata = pd.DataFrame(metadata_rows)
    cycle_summary = pd.concat(cycle_parts, ignore_index=True)
    validate_processed_tables(metadata, cycle_summary, require_labels=True)
    splits = make_split_assignments(metadata, SplitConfig(seed=args.seed))
    validate_split_assignments(metadata, splits, require_train_labels=True)

    metadata.to_csv(out / "cell_metadata.csv", index=False)
    cycle_summary.to_csv(out / "cycle_summary.csv", index=False)
    splits.to_csv(out / "splits.csv", index=False)
    write_json(out / "qc_summary.json", qc_summary(metadata, cycle_summary, splits))
    print(f"Wrote demo dataset to {out}")


if __name__ == "__main__":
    main()
