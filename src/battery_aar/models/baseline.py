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

NON_FEATURE_COLUMNS = {"cell_id", "cycle_life", "split", "protocol_readable", "batch_id"}
SUPPORTED_MODEL_KINDS = ("dummy_mean", "ridge", "random_forest", "gradient_boosting")


@dataclass
class TrainedBaseline:
    """Container for a trained model and its feature columns."""

    model: Pipeline
    feature_columns: list[str]
    model_kind: str


def select_feature_columns(features: pd.DataFrame) -> list[str]:
    """Select numeric feature columns, excluding IDs, labels, and split metadata."""

    columns: list[str] = []
    for col in features.columns:
        if col in NON_FEATURE_COLUMNS:
            continue
        values = pd.to_numeric(features[col], errors="coerce")
        if values.notna().any():
            columns.append(col)
    return columns


def make_regressor(kind: str = "random_forest", *, seed: int = 42, **kwargs: Any) -> Pipeline:
    """Create a baseline regression pipeline."""

    kind = kind.lower()
    if kind == "dummy_mean":
        regressor = DummyRegressor(strategy="mean")
        steps = [("imputer", SimpleImputer(strategy="median")), ("regressor", regressor)]
    elif kind == "ridge":
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
    feature_columns = select_feature_columns(features)
    if not feature_columns:
        raise ValueError("no numeric feature columns found")
    model = make_regressor(model_kind, seed=seed, **model_kwargs)
    X = train_df[feature_columns].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(train_df["cycle_life"], errors="coerce")
    mask = y.notna()
    if mask.sum() == 0:
        raise ValueError("no finite training labels found")
    model.fit(X.loc[mask], y.loc[mask].to_numpy(dtype=float))
    return TrainedBaseline(model=model, feature_columns=feature_columns, model_kind=model_kind)


def predict(trained: TrainedBaseline, features: pd.DataFrame) -> np.ndarray:
    """Generate predictions for a trained baseline."""

    X = features[trained.feature_columns].apply(pd.to_numeric, errors="coerce")
    return np.asarray(trained.model.predict(X), dtype=float)
