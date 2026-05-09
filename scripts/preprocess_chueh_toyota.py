#!/usr/bin/env python
"""Preprocess public Chueh/Toyota fast-charging data into normalized tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from battery_aar.data.matr_io import load_raw_batches
from battery_aar.data.schema import (
    qc_summary,
    validate_processed_tables,
    validate_split_assignments,
)
from battery_aar.data.split import SplitConfig, make_split_assignments
from battery_aar.utils.io import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/chueh_toyota_fast_charge"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/chueh_toyota_fast_charge"))
    parser.add_argument("--max-cells-per-batch", type=int, default=None)
    parser.add_argument("--first-n-cycles", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-mode",
        type=str,
        default="random",
        choices=["random", "batch", "protocol", "protocol_cluster", "leave_one_batch_out"],
        help="Default random split is for smoke tests; scientific claims require group/transfer splits.",
    )
    parser.add_argument("--split-group-column", type=str, default=None)
    parser.add_argument("--leave-one-batch-id", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = ensure_dir(args.out)
    loaded = load_raw_batches(
        args.raw_dir,
        max_cells_per_batch=args.max_cells_per_batch,
        first_n_cycles=args.first_n_cycles,
    )
    validate_processed_tables(loaded.metadata, loaded.cycle_summary, require_labels=False)
    labels = pd.to_numeric(loaded.metadata["cycle_life"], errors="coerce")
    labeled_metadata = loaded.metadata.loc[labels.notna()].copy()
    if labeled_metadata.empty:
        raise ValueError(
            "No cycle_life labels found. For BEEP JSON exports, labels are only derived "
            "when a cell visibly crosses the 80% capacity threshold."
        )
    split_mode = "leave_one_batch_out" if args.leave_one_batch_id else args.split_mode
    splits = make_split_assignments(
        labeled_metadata,
        SplitConfig(
            mode=split_mode,
            seed=args.seed,
            group_column=args.split_group_column,
            leave_one_group_value=args.leave_one_batch_id,
        ),
    )
    unlabeled_cells = loaded.metadata.loc[labels.isna(), ["cell_id"]].copy()
    if not unlabeled_cells.empty:
        unlabeled_cells["split"] = "unlabeled"
        splits = pd.concat([splits, unlabeled_cells], ignore_index=True)
    validate_split_assignments(loaded.metadata, splits, require_train_labels=True)

    loaded.metadata.to_csv(out / "cell_metadata.csv", index=False)
    loaded.cycle_summary.to_csv(out / "cycle_summary.csv", index=False)
    splits.to_csv(out / "splits.csv", index=False)
    write_json(out / "qc_summary.json", qc_summary(loaded.metadata, loaded.cycle_summary, splits))
    print(f"Wrote processed dataset to {out}")


if __name__ == "__main__":
    main()
