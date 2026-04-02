# src/pipeline/run_pipeline.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


import pandas as pd

# Your existing modules
from src.data.load_data import load_raw_ckd
from src.data.preprocess import standardize_columns, coerce_types
from src.data.validate_labs import validate_labs
from src.features.derive import add_derived_features


# The contract
from src.data.contracts import (
    TARGET_COL,
    MODEL_REQUIRED_BASE_FEATURES,
    MODEL_REQUIRED_ENHANCED_FEATURES,
    missing_columns,
    assert_required_columns,
)


@dataclass(frozen=True)
class PipelineResult:
    df_raw: pd.DataFrame
    df_clean: pd.DataFrame
    df_valid: pd.DataFrame
    issues: Dict[str, Any]
    X_base: Optional[pd.DataFrame]
    X_enhanced: Optional[pd.DataFrame]


# -------------------------
# 1) load_raw(path) -> df
# -------------------------
def load_raw(path: str) -> pd.DataFrame:
    """
    Single loader wrapper.
    Uses your existing load_data.py so you don't duplicate logic.
    """
    return load_raw_ckd(path)   # ✅ FIXED (was calling itself)


# -------------------------
# 2) preprocess(df) -> df_clean
# -------------------------
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Canonical preprocessing:
    - standardize/rename columns to internal schema
    - coerce types
    """
    df2 = standardize_columns(df)
    df2 = coerce_types(df2)
    return df2


# -------------------------
# 3) validate(df_clean) -> (df_valid, issues)
# -------------------------
def validate(df_clean: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Canonical validation:
    - biological plausibility checks
    - returns cleaned df + validation report
    """
    report = validate_labs(df_clean)
    return df_clean, report


# -------------------------
# 4) build_features(df_valid, mode) -> X
# -------------------------
def build_features(df_valid: pd.DataFrame, mode: str = "base") -> pd.DataFrame:
    """
    Returns a feature matrix X as a DataFrame with the exact required columns.

    IMPORTANT:
    - This function does NOT do fancy encoding yet (one-hot/scaling).
      It only selects the correct columns so stage separation is enforced.
    """
    mode = mode.lower().strip()
    if mode not in {"base", "enhanced"}:
        raise ValueError(f"mode must be 'base' or 'enhanced', got: {mode}")

    required = MODEL_REQUIRED_BASE_FEATURES if mode == "base" else MODEL_REQUIRED_ENHANCED_FEATURES

    # Contract enforcement: fail fast if the columns aren't present
    assert_required_columns(df_valid, required, context=f"build_features:{mode}")

    return df_valid[required].copy()


# -------------------------
# 5) run_pipeline(path) -> PipelineResult
# -------------------------
def run_pipeline(
    path: str,
    *,
    build_base: bool = True,
    build_enhanced: bool = True,
    require_target: bool = False,
) -> PipelineResult:
    """
    The ONLY entry point you use everywhere:
    - EDA
    - training
    - inference
    - API

    Controls:
    - build_base/enhanced decide which X matrices to return
    - require_target=True for training, False for inference
    """
    df_raw = load_raw(path)
    df_clean = preprocess(df_raw)
    df_clean = add_derived_features(df_clean)

    # If training, enforce target presence early
    if require_target:
        assert_required_columns(df_clean, [TARGET_COL], context="run_pipeline:target")

    df_valid, issues = validate(df_clean)

    X_base = build_features(df_valid, mode="base") if build_base else None

    if build_enhanced:
        enhanced_missing = missing_columns(df_valid.columns, MODEL_REQUIRED_ENHANCED_FEATURES)
        X_enhanced = None if enhanced_missing else build_features(df_valid, mode="enhanced")
    else:
        X_enhanced = None

    return PipelineResult(
        df_raw=df_raw,
        df_clean=df_clean,
        df_valid=df_valid,
        issues=issues,
        X_base=X_base,
        X_enhanced=X_enhanced,
    )
