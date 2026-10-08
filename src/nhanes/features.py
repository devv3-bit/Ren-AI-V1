# src/nhanes/features.py
"""
Feature construction (Step 6).

  blood tests      17 serum / whole-blood analytes (BLOOD_TEST_FEATURES)
  vitals / demo    age, sex, systolic + diastolic BP, BMI, diabetes
  engineered       egfr_cr (CKD-EPI 2021), bun_creatinine_ratio
  missingness      indicator columns are added inside the sklearn imputer

Urine variables (urine_albumin, urine_creatinine, acr) define the label and
are never features. eGFR is both a feature and part of the label, which is
why the early-subgroup AUC is reported separately.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.features.build_features import _safe_log1p
from src.features.derive import _safe_div
from src.nhanes.constants import (
    BLOOD_TEST_FEATURES,
    ENGINEERED_FEATURES,
    HARMONIZED_FEATURES,
    LOG_FEATURES,
    MODEL_FEATURES,
    TEST_CYCLES,
    TRAIN_CYCLES,
    URINE_COLS,
    VITALS_DEMO_FEATURES,
)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add bun_creatinine_ratio (egfr_cr is added by src.nhanes.label.add_egfr)."""
    if "bun" not in df.columns or "serum_creatinine" not in df.columns:
        raise ValueError("add_engineered_features: bun and serum_creatinine are required")
    return df.assign(bun_creatinine_ratio=_safe_div(df["bun"], df["serum_creatinine"]))


def build_X(
    df: pd.DataFrame,
    feature_cols: Sequence[str] = MODEL_FEATURES,
    log_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Numeric feature frame in a fixed column order; heavy-tailed labs log1p-transformed."""
    feature_cols = list(feature_cols)
    leak = sorted(set(feature_cols) & set(URINE_COLS))
    if leak:
        raise ValueError(f"urine-based columns can never be features: {leak}")
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"build_X: missing feature columns {missing}")

    X = df.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce")
    to_log = [c for c in (LOG_FEATURES if log_cols is None else log_cols) if c in X.columns]
    if to_log:
        X = X.assign(**{c: _safe_log1p(X[c].to_numpy(dtype=float)) for c in to_log})
    return X.astype(float)


def split_train_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Temporal split: 2005-2016 cycles train, 2017-2020 test."""
    cycle = df["cycle"].astype(str)
    train = df[cycle.isin(TRAIN_CYCLES)].reset_index(drop=True)
    test = df[cycle.isin(TEST_CYCLES)].reset_index(drop=True)
    unknown = sorted(set(cycle.unique()) - set(TRAIN_CYCLES) - set(TEST_CYCLES))
    if unknown:
        raise ValueError(f"cycles not assigned to a split: {unknown}")
    return train, test


def feature_inventory() -> Dict[str, object]:
    """The exact counts quoted in the write-up and the claims sheet."""
    return {
        "n_blood_tests": len(BLOOD_TEST_FEATURES),
        "blood_tests": list(BLOOD_TEST_FEATURES),
        "n_vitals_demographics": len(VITALS_DEMO_FEATURES),
        "vitals_demographics": list(VITALS_DEMO_FEATURES),
        "n_engineered_scores": len(ENGINEERED_FEATURES),
        "engineered_scores": list(ENGINEERED_FEATURES),
        "n_model_features": len(MODEL_FEATURES),
        "log_transformed": list(LOG_FEATURES),
        "n_harmonized_uci_features": len(HARMONIZED_FEATURES),
        "harmonized_uci_features": list(HARMONIZED_FEATURES),
        "never_features": list(URINE_COLS),
    }
