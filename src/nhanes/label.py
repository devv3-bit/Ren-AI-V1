# src/nhanes/label.py
"""
KDIGO-style CKD label (Step 5), single visit.

  egfr_cr        CKD-EPI 2021 creatinine eGFR using each participant's recorded sex
  egfr_low       eGFR < 60 mL/min/1.73 m2            (KDIGO G3a or worse)
  albuminuria    ACR >= 30 mg/g                       (KDIGO A2 or worse)
  ckd            1 if egfr_low OR albuminuria else 0
  early_subgroup eGFR >= 60: CKD here can only come from albuminuria
                 (G1-G2 with A2-A3), i.e. early-stage disease

Limitation: NHANES is a single visit, so the >= 3 month chronicity criterion
cannot be verified; this is a cross-sectional proxy for CKD.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from src.nhanes.constants import (
    ACR_CKD_THRESHOLD,
    EARLY_SUBGROUP_COL,
    EGFR_CKD_THRESHOLD,
    TARGET_COL,
)
from src.nhanes.egfr import egfr_cr_vectorized


def add_egfr(df: pd.DataFrame) -> pd.DataFrame:
    """Add egfr_cr computed from serum_creatinine, age and the recorded sex."""
    for col in ("serum_creatinine", "age", "sex"):
        if col not in df.columns:
            raise ValueError(f"add_egfr: missing column {col!r}")
    return df.assign(egfr_cr=egfr_cr_vectorized(df["serum_creatinine"], df["age"], df["sex"]))


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add egfr_low, albuminuria, ckd (int) and early_subgroup (bool)."""
    if "egfr_cr" not in df.columns or "acr" not in df.columns:
        raise ValueError("add_labels: egfr_cr and acr are required (run add_egfr first)")
    egfr = pd.to_numeric(df["egfr_cr"], errors="coerce")
    acr = pd.to_numeric(df["acr"], errors="coerce")
    n_bad = int((egfr.isna() | acr.isna()).sum())
    if n_bad:
        raise ValueError(f"add_labels: {n_bad} rows have missing eGFR or ACR; apply the cohort filters first")

    egfr_low = egfr < EGFR_CKD_THRESHOLD
    albuminuria = acr >= ACR_CKD_THRESHOLD
    return df.assign(
        egfr_low=egfr_low.to_numpy(),
        albuminuria=albuminuria.to_numpy(),
        **{TARGET_COL: (egfr_low | albuminuria).astype(int).to_numpy()},
        **{EARLY_SUBGROUP_COL: (egfr >= EGFR_CKD_THRESHOLD).to_numpy()},
    )


def prevalence_summary(df: pd.DataFrame) -> Dict[str, float]:
    """Overall and early-subgroup prevalence plus the label composition."""
    y = df[TARGET_COL].astype(int)
    early = df[EARLY_SUBGROUP_COL].astype(bool)
    return {
        "n": int(len(df)),
        "n_ckd": int(y.sum()),
        "prevalence": float(y.mean()),
        "n_egfr_low": int(df["egfr_low"].sum()),
        "n_albuminuria": int(df["albuminuria"].sum()),
        "n_both": int((df["egfr_low"] & df["albuminuria"]).sum()),
        "n_early_subgroup": int(early.sum()),
        "n_ckd_early_subgroup": int(y[early].sum()),
        "prevalence_early_subgroup": float(y[early].mean()) if early.any() else float("nan"),
    }


def prevalence_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for key, part in df.groupby(group_col, sort=False):
        s = prevalence_summary(part)
        rows.append({group_col: key, **s})
    return pd.DataFrame(rows)
