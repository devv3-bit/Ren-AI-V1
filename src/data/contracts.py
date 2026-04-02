# src/data/contracts.py
"""
Data Contract: the single source of truth for:
- canonical internal column names
- feature groups (numeric/binary/categorical/enhanced)
- required feature sets for each model stage
- enforcement helpers (fail fast)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Iterable

from src.config import (
    I_SCHEMA,
    NUMERIC_COLS,
    BINARY_COLS,
    OPTIONAL_ENHANCED_BIOMARKERS,
)

# -------------------------
# Canonical schema elements
# -------------------------

TARGET_COL = "ckd_class"

# Columns that exist in CKD.csv but are not numeric/binary
# (Based on your I_SCHEMA: rbc, pc, pcc, ba, appet are categorical strings)
CATEGORICAL_COLS: List[str] = [
    "urine_red_blood_cells",
    "pus_cell",
    "pus_cell_clumps",
    "bacteria",
    "appetite",
]

# Enhanced biomarkers (not in CKD.csv)
ENHANCED_BIOMARKERS: List[str] = list(OPTIONAL_ENHANCED_BIOMARKERS.keys())

# All internal columns (features only) expected in the "base world"
INTERNAL_BASE_FEATURES: List[str] = (
    list(NUMERIC_COLS) + list(BINARY_COLS) + list(CATEGORICAL_COLS)
)

# -------------------------
# Two-stage model contract
# -------------------------

# Stage 1 (base model): must work with only CKD.csv-like inputs
MODEL_REQUIRED_BASE_FEATURES: List[str] = list(INTERNAL_BASE_FEATURES)

# Stage 2 (enhanced model): base + enhanced biomarkers
MODEL_REQUIRED_ENHANCED_FEATURES: List[str] = (
    list(INTERNAL_BASE_FEATURES) + list(ENHANCED_BIOMARKERS)
)

# -------------------------
# Enforcement helpers
# -------------------------

def missing_columns(df_cols: Iterable[str], required: Iterable[str]) -> Set[str]:
    cols = set(df_cols)
    req = set(required)
    return req - cols

def assert_required_columns(df, required: Iterable[str], *, context: str = "") -> None:
    miss = missing_columns(df.columns, required)
    if miss:
        where = f" ({context})" if context else ""
        raise ValueError(
            f"Data contract violation{where}: missing columns: {sorted(miss)}"
        )

def assert_no_unknown_columns(df, allowed: Iterable[str], *, context: str = "") -> None:
    cols = set(df.columns)
    allow = set(allowed)
    extra = cols - allow
    if extra:
        where = f" ({context})" if context else ""
        raise ValueError(
            f"Data contract violation{where}: unknown columns: {sorted(extra)}"
        )

def base_allowed_columns() -> Set[str]:
    return set(MODEL_REQUIRED_BASE_FEATURES) | {TARGET_COL}

def enhanced_allowed_columns() -> Set[str]:
    return set(MODEL_REQUIRED_ENHANCED_FEATURES) | {TARGET_COL}
