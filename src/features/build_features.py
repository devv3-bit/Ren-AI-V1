"""
Feature Engineering (Base Model)

Goal:
- Take df_valid (clean + validated dataframe) and produce:
  - X: numeric feature matrix ready for ML
  - y: binary target vector (0/1) if target exists / requested
- Apply transformations decided from EDA:
  - Impute missing numeric values (median) + missingness indicators
  - Log-transform heavy-tailed features (log1p after clipping at 0)
  - Winsorize/cap extreme outliers (quantile clipping)
  - Standardize numeric features (z-score)
- Keep this file "model training ready" (sklearn Pipeline).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------
# Imports from your project
# ---------------------------------------------------------------------
try:
    # Contracts define the official target column + (often) required features.
    from src.data.contracts import TARGET_COL, assert_required_columns
except Exception:  # pragma: no cover
    TARGET_COL = "ckd_class"

    def assert_required_columns(df, required, context=""):
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns {missing} ({context})")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _to_binary_target(series: pd.Series) -> np.ndarray:
    """
    Convert target to {0,1}.
    Accepts values like: "ckd", "notckd", "1", "0", True/False, etc.
    """
    s = series.astype(str).str.strip().str.lower()

    positive = {"ckd", "1", "true", "yes", "y", "positive"}
    negative = {"notckd", "0", "false", "no", "n", "negative"}

    y = np.full(shape=(len(s),), fill_value=np.nan, dtype=float)
    y[s.isin(positive)] = 1.0
    y[s.isin(negative)] = 0.0

    if np.isnan(y).any():
        bad = series[pd.isna(pd.Series(y))].unique()[:10]
        raise ValueError(f"Unrecognized target labels in {TARGET_COL}: {bad}")

    return y.astype(int)


def _safe_log1p(X: np.ndarray) -> np.ndarray:
    """
    log1p requires X >= -1; for lab features we clamp at 0 before log1p.
    """
    X = np.asarray(X, dtype=float)
    X = np.where(np.isfinite(X), X, np.nan)
    X = np.maximum(X, 0.0)
    return np.log1p(X)


@dataclass(frozen=True)
class FeatureBuildResult:
    X_df: pd.DataFrame
    y: Optional[np.ndarray]
    feature_names: List[str]
    preprocessor: Pipeline


# ---------------------------------------------------------------------
# Default feature policy (from your EDA summary)
# ---------------------------------------------------------------------
DEFAULT_BASE_FEATURES: List[str] = [
    "age",
    "blood_pressure",
    "specific_gravity",
    "albumin",
    "blood_glucose_random",
    "blood_urea",
    "serum_creatinine",
    "sodium",
    "hemoglobin",
    "packed_cell_volume",
    "red_blood_cell_count",
    # derived (if present)
    "egfr_cr",
    "urea_creatinine_ratio",
]

# Heavy-tailed -> log transform recommended
DEFAULT_LOG_COLS: List[str] = [
    "serum_creatinine",
    "blood_urea",
    "blood_glucose_random",
    "urea_creatinine_ratio",
]

DEFAULT_LOG_FEATURES = DEFAULT_LOG_COLS


def get_base_X_df(
    df_valid: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Return raw feature frame (numeric, log-transformed) without fitting anything.
    """
    if feature_cols is None:
        feature_cols = [c for c in DEFAULT_BASE_FEATURES if c in df_valid.columns]
    else:
        feature_cols = [c for c in feature_cols if c in df_valid.columns]

    assert_required_columns(df_valid, list(feature_cols), context="get_base_X_df:features")

    X_df = df_valid.loc[:, list(feature_cols)].copy()
    for c in X_df.columns:
        X_df[c] = pd.to_numeric(X_df[c], errors="coerce")

    log_cols = [c for c in DEFAULT_LOG_COLS if c in X_df.columns]
    if log_cols:
        X_df[log_cols] = _safe_log1p(X_df[log_cols].to_numpy(dtype=float))

    return X_df


def get_y(
    df_valid: pd.DataFrame,
    *,
    require_target: bool = True,
    target_col: str = TARGET_COL,
) -> Optional[np.ndarray]:
    if require_target:
        assert_required_columns(df_valid, [target_col], context="get_y:target")
        return _to_binary_target(df_valid[target_col])

    return None


def make_preprocessor(*, add_missing_indicators: bool = True) -> Pipeline:
    imputer = SimpleImputer(strategy="median", add_indicator=add_missing_indicators)
    scaler = StandardScaler()
    return Pipeline(
        steps=[
            ("imputer", imputer),
            ("scaler", scaler),
        ]
    )


def build_base_features(
    df_valid: pd.DataFrame,
    *,
    feature_cols: Optional[Sequence[str]] = None,
    require_target: bool = True,
    target_col: str = TARGET_COL,
    add_missing_indicators: bool = True,
) -> FeatureBuildResult:
    """
    Build ML-ready (X, y) for the BASE model.

    Parameters
    ----------
    df_valid:
        Output of your pipeline validation stage.
    feature_cols:
        Columns to use as features. Defaults to DEFAULT_BASE_FEATURES filtered to columns that exist.
    require_target:
        If True, raises if target_col missing. If False, y=None.
    target_col:
        Target label column.
    add_missing_indicators:
        If True, add extra indicator columns for missing values (important when missingness is MNAR).

    Returns
    -------
    FeatureBuildResult:
        X_df (pd.DataFrame), y (np.ndarray or None), feature_names, unfitted preprocessing Pipeline
    """
    X_df = get_base_X_df(df_valid, feature_cols=feature_cols)
    y = get_y(df_valid, require_target=require_target, target_col=target_col)

    preprocessor = make_preprocessor(add_missing_indicators=add_missing_indicators)

    return FeatureBuildResult(
        X_df=X_df,
        y=y,
        feature_names=list(X_df.columns),
        preprocessor=preprocessor,
    )
