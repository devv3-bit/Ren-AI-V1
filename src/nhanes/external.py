# src/nhanes/external.py
"""
External validation data (Step 10): the UCI CKD hospital dataset
(data/raw/CKD.csv, 400 rows, Indian hospital patients).

Loaded with the existing v1 pipeline (src.pipeline.run_pipeline) and
harmonised to the NHANES feature names:
  bun                 = blood_urea / 2.14   (urea mg/dL -> BUN mg/dL)
  blood_pressure      already diastolic (UCI `bp`)
  packed_cell_volume  UCI `pcv` (same canonical name)
  diabetes_mellitus   "yes"/"no" strings -> 1/0
  sex                 not recorded in UCI: eGFR is computed under an assumed
                      sex ("male" = the repo's v1 fallback; "female" is a
                      sensitivity analysis)
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.features.build_features import get_y
from src.nhanes.constants import UCI_PATH, UREA_TO_BUN_DIVISOR
from src.nhanes.features import add_engineered_features
from src.nhanes.label import add_egfr
from src.pipeline.run_pipeline import run_pipeline

SEX_ASSUMPTIONS = ("male", "female")


def yes_no_to_float(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip().str.lower()
    return s.map({"yes": 1.0, "no": 0.0}).astype(float)


def load_uci_harmonized(sex_assumption: str = "male", path: Path = UCI_PATH) -> pd.DataFrame:
    """UCI rows in the NHANES canonical schema (target column ckd_class kept)."""
    if sex_assumption not in SEX_ASSUMPTIONS:
        raise ValueError(f"sex_assumption must be one of {SEX_ASSUMPTIONS}, got {sex_assumption!r}")
    df = run_pipeline(str(path), require_target=True).df_valid
    out = df.assign(
        bun=pd.to_numeric(df["blood_urea"], errors="coerce") / UREA_TO_BUN_DIVISOR,
        diabetes_mellitus=yes_no_to_float(df["diabetes_mellitus"]),
        sex=sex_assumption,
    )
    return add_engineered_features(add_egfr(out))


def uci_target(df: pd.DataFrame) -> np.ndarray:
    return get_y(df, require_target=True)


def feature_availability(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    rows = [
        {
            "feature": f,
            "n_nonmissing": int(pd.to_numeric(df[f], errors="coerce").notna().sum()) if f in df.columns else 0,
            "frac_missing": float(pd.to_numeric(df[f], errors="coerce").isna().mean()) if f in df.columns else 1.0,
        }
        for f in features
    ]
    return pd.DataFrame(rows)
