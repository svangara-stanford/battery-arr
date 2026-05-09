#!/usr/bin/env python
"""Preprocess public Chueh/Toyota fast-charging data into normalized tables."""

from __future__ import annotations

import argparse
from pathlib import Path

from battery_aar.data.matr_io import load_raw_batches
from battery_aar.data.schema import qc_summary, validate_processed_tables
from battery_aar.data.split import SplitConfig, make_split_assignments
from battery_aar.utils.io import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/chueh_toyota_fast_charge"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/chueh_toyota_fast_charge"))
    parser.add_argument("--max-cells-per-batch", type=int, default=None)
    parser.add_argument("--first-n-cycles", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-group-column", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = ensure_dir(args.out)
    loaded = load_raw_batches(
        args.raw_dir,
        max_cells_per_batch=args.max_cells_per_batch,
        first_n_cycles=args.first_n_cycles,
    )
    validate_processed_tables(loaded.metadata, loaded.cycle_summary, require_labels=True)
    splits = make_split_assignments(
        loaded.metadata,
        SplitConfig(seed=args.seed, group_column=args.split_group_column),
    )

    loaded.metadata.to_csv(out / "cell_metadata.csv", index=False)
    loaded.cycle_summary.to_csv(out / "cycle_summary.csv", index=False)
    splits.to_csv(out / "splits.csv", index=False)
    write_json(out / "qc_summary.json", qc_summary(loaded.metadata, loaded.cycle_summary))
    print(f"Wrote processed dataset to {out}")


if __name__ == "__main__":
    main()
