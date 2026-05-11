"""Scikit-learn baseline models for early-cycle battery lifetime prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NON_FEATURE_COLUMNS = {
    "cell_id",
    "cycle_life",
    "split",
    "protocol_readable",
    "batch_id",
    "paper_curve_features_available",
    "paper_missing_curve_feature_reason",
}
LOG_LIFE_MODEL_KINDS = ("paper_ridge_loglife", "paper_linear_loglife")
SUPPORTED_MODEL_KINDS = (
    "dummy_mean",
    "ridge",
    "random_forest",
    "gradient_boosting",
    *LOG_LIFE_MODEL_KINDS,
)


@dataclass
class TrainedBaseline:
    """Container for a trained model and its feature columns."""

    model: Pipeline
    feature_columns: list[str]
    model_kind: str
    target_transform: str = "identity"
    dropped_feature_columns: list[str] | None = None


def select_feature_columns(features: pd.DataFrame) -> list[str]:
    """Select numeric feature columns, excluding IDs, labels, and split metadata."""

    columns: list[str] = []
    for col in _candidate_feature_columns(features):
        values = pd.to_numeric(features[col], errors="coerce")
        if values.notna().any():
            columns.append(col)
    return columns


def _candidate_feature_columns(features: pd.DataFrame) -> list[str]:
    """Return columns that are intended as model inputs before missingness checks."""

    return [col for col in features.columns if col not in NON_FEATURE_COLUMNS]


def make_regressor(kind: str = "random_forest", *, seed: int = 42, **kwargs: Any) -> Pipeline:
    """Create a baseline regression pipeline."""

    kind = kind.lower()
    if kind == "dummy_mean":
        regressor = DummyRegressor(strategy="mean")
        steps = [("imputer", SimpleImputer(strategy="median")), ("regressor", regressor)]
    elif kind in {"ridge", "paper_ridge_loglife", "paper_linear_loglife"}:
        alpha = float(kwargs.get("alpha", 1.0))
        regressor = Ridge(alpha=alpha, random_state=seed)
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", regressor),
        ]
    elif kind == "random_forest":
        regressor = RandomForestRegressor(
            n_estimators=int(kwargs.get("n_estimators", 200)),
            max_depth=kwargs.get("max_depth", None),
            min_samples_leaf=int(kwargs.get("min_samples_leaf", 1)),
            random_state=seed,
            n_jobs=int(kwargs.get("n_jobs", -1)),
        )
        steps = [("imputer", SimpleImputer(strategy="median")), ("regressor", regressor)]
    elif kind == "gradient_boosting":
        regressor = GradientBoostingRegressor(
            n_estimators=int(kwargs.get("n_estimators", 200)),
            max_depth=int(kwargs.get("max_depth", 3)),
            learning_rate=float(kwargs.get("learning_rate", 0.05)),
            random_state=seed,
        )
        steps = [("imputer", SimpleImputer(strategy="median")), ("regressor", regressor)]
    else:
        raise ValueError(f"Unknown model kind {kind!r}; expected one of {SUPPORTED_MODEL_KINDS}")
    return Pipeline(steps)


def train_baseline(
    features: pd.DataFrame,
    *,
    split_column: str = "split",
    model_kind: str = "random_forest",
    seed: int = 42,
    feature_columns: list[str] | None = None,
    **model_kwargs: Any,
) -> TrainedBaseline:
    """Train a baseline model on rows assigned to the train split."""

    if "cycle_life" not in features.columns:
        raise ValueError("features must include cycle_life labels")
    if split_column not in features.columns:
        raise ValueError(f"features must include {split_column!r}")
    train_df = features[features[split_column] == "train"].copy()
    if train_df.empty:
        raise ValueError("no training rows found")
    candidate_columns = feature_columns or _candidate_feature_columns(features)
    if not candidate_columns:
        raise ValueError("no candidate feature columns found")
    missing_features = set(candidate_columns).difference(features.columns)
    if missing_features:
        raise ValueError(f"requested feature columns missing: {sorted(missing_features)}")
    all_numeric = features[candidate_columns].apply(pd.to_numeric, errors="coerce")
    train_numeric = train_df[candidate_columns].apply(pd.to_numeric, errors="coerce")
    dropped_feature_columns = [
        col
        for col in candidate_columns
        if not all_numeric[col].notna().any() or not train_numeric[col].notna().any()
    ]
    feature_columns = [col for col in candidate_columns if col not in dropped_feature_columns]
    if not feature_columns:
        raise ValueError("all candidate feature columns are missing in the training split")
    model = make_regressor(model_kind, seed=seed, **model_kwargs)
    X = train_df[feature_columns].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(train_df["cycle_life"], errors="coerce")
    target_transform = "log10_cycle_life" if model_kind in LOG_LIFE_MODEL_KINDS else "identity"
    mask = y.notna() & (y > 0)
    if mask.sum() == 0:
        raise ValueError("no finite training labels found")
    y_train = y.loc[mask].to_numpy(dtype=float)
    if target_transform == "log10_cycle_life":
        y_train = np.log10(y_train)
    model.fit(X.loc[mask], y_train)
    return TrainedBaseline(
        model=model,
        feature_columns=feature_columns,
        model_kind=model_kind,
        target_transform=target_transform,
        dropped_feature_columns=dropped_feature_columns,
    )


def predict(trained: TrainedBaseline, features: pd.DataFrame) -> np.ndarray:
    """Generate predictions for a trained baseline."""

    raw = predict_model_output(trained, features)
    if trained.target_transform == "log10_cycle_life":
        return np.asarray(10.0**raw, dtype=float)
    return raw


def predict_model_output(trained: TrainedBaseline, features: pd.DataFrame) -> np.ndarray:
    """Generate predictions in the model target space."""

    X = features[trained.feature_columns].apply(pd.to_numeric, errors="coerce")
    return np.asarray(trained.model.predict(X), dtype=float)


def predict_log10_cycle_life(trained: TrainedBaseline, features: pd.DataFrame) -> np.ndarray:
    """Generate log10(cycle_life) predictions for log-life baselines."""

    raw = predict_model_output(trained, features)
    if trained.target_transform == "log10_cycle_life":
        return raw
    cycle_pred = np.asarray(raw, dtype=float)
    return np.where(cycle_pred > 0, np.log10(cycle_pred), np.nan)
