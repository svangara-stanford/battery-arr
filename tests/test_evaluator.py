from __future__ import annotations

import pandas as pd
import pytest

from battery_aar.evaluation.evaluator import evaluate_predictions


def test_evaluate_predictions_rejects_label_column_in_submission() -> None:
    truth = pd.DataFrame({"cell_id": ["a"], "cycle_life": [100.0]})
    pred = pd.DataFrame({"cell_id": ["a"], "y_pred": [101.0], "cycle_life": [100.0]})
    with pytest.raises(ValueError, match="only cell_id and y_pred"):
        evaluate_predictions(truth, pred)


def test_evaluate_predictions_scores_minimal_submission() -> None:
    truth = pd.DataFrame({"cell_id": ["a", "b"], "cycle_life": [100.0, 200.0]})
    pred = pd.DataFrame({"cell_id": ["a", "b"], "y_pred": [110.0, 190.0]})
    metrics = evaluate_predictions(truth, pred)
    assert metrics["n"] == 2
    assert metrics["rmse"] == 10.0
