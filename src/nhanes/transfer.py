# src/nhanes/transfer.py
"""
v3 transfer model for the external validation (a disclosed SECOND look at UCI).

Pre-specified after the post-hoc diagnosis in reports/nhanes/external_diagnostics.json
and before UCI was evaluated again. Changes relative to the v2 harmonised Model A:

  1. No missing-value indicators. Missingness is site-specific and does not
     transfer; in v2 the standardised "missing potassium" indicator alone
     shifted UCI patients by -55 log-odds.
  2. Features are clipped to the NHANES-train 0.1-99.9 percentile range before
     scaling (RangeClipper), so hospital-range values (creatinine up to 76 mg/dL,
     z = 26 in v2) cannot extrapolate a linear model.
  3. A gradient-boosting candidate on the same 14 shared features (trees do not
     extrapolate and handle missing values natively).

Candidates are still selected by 5-fold CV ROC-AUC on NHANES train only.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.nhanes.constants import SEED

CLIP_LOWER_QUANTILE = 0.001
CLIP_UPPER_QUANTILE = 0.999


class RangeClipper(BaseEstimator, TransformerMixin):
    """Clip every column to the [lower, upper] quantiles seen at fit time; NaN passes through."""

    def __init__(self, lower: float = CLIP_LOWER_QUANTILE, upper: float = CLIP_UPPER_QUANTILE):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if not 0.0 <= self.lower < self.upper <= 1.0:
            raise ValueError("require 0 <= lower < upper <= 1")
        self.lower_ = np.nanquantile(X, self.lower, axis=0)
        self.upper_ = np.nanquantile(X, self.upper, axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} columns, got {X.shape[1]}")
        return np.clip(X, self.lower_, self.upper_)  # np.clip propagates NaN

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def make_transfer_logistic(C: float, clip: bool = True) -> Pipeline:
    """Median imputation (no indicators) -> optional train-range clipping -> scaler -> balanced L2 logistic."""
    steps = []
    if clip:
        steps.append(("clip", RangeClipper()))
    steps += [
        ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=SEED)),
    ]
    return Pipeline(steps=steps)
