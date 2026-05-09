#!/usr/bin/env python
"""Evaluate a prediction submission against a truth CSV.

For the local demo, `truth` is stored on disk. In the future benchmark, this
script becomes the hidden evaluator path where labels are not mounted into agent
sandboxes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from battery_aar.evaluation.evaluator import evaluate_predictions
from battery_aar.utils.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", type=Path, required=True, help="CSV with cell_id,cycle_life")
    parser.add_argument("--pred", type=Path, required=True, help="CSV with cell_id,y_pred")
    parser.add_argument("--out", type=Path, required=True, help="Output metrics JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    truth = pd.read_csv(args.truth)
    pred = pd.read_csv(args.pred)
    metrics = evaluate_predictions(truth, pred)
    write_json(args.out, metrics)
    print(metrics)


if __name__ == "__main__":
    main()
