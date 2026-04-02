# src/features/derive.py
from __future__ import annotations

from typing import Literal, Union
import numpy as np
import pandas as pd

Sex = Literal["male", "female", "m", "f"]

# -------------------------
# helpers
# -------------------------
def _norm_sex(sex: Union[str, Sex]) -> str:
    s = str(sex).strip().lower()
    if s in {"m", "male"}:
        return "male"
    if s in {"f", "female"}:
        return "female"
    raise ValueError(f"sex must be male/female (or m/f). Got: {sex!r}")


def _safe_div(numer, denom):
    numer = pd.to_numeric(numer, errors="coerce")
    denom = pd.to_numeric(denom, errors="coerce")
    out = numer / denom.replace({0: np.nan})
    return out


# -------------------------
# eGFR equations (adult CKD-EPI 2021 race-free)
# Units: creatinine mg/dL, cystatin C mg/L, age years
# -------------------------
def egfr_cr(creatinine, age, sex: Sex) -> float:
    sex_n = _norm_sex(sex)
    Scr = float(creatinine)
    Age = float(age)

    k = 0.7 if sex_n == "female" else 0.9
    a = -0.241 if sex_n == "female" else -0.302
    sex_factor = 1.012 if sex_n == "female" else 1.0

    x = Scr / k
    return float(
        142.0
        * (min(x, 1.0) ** a)
        * (max(x, 1.0) ** -1.200)
        * (0.9938 ** Age)
        * sex_factor
    )


def egfr_cys(cystatin_c, age, sex: Sex) -> float:
    sex_n = _norm_sex(sex)
    Scys = float(cystatin_c)
    Age = float(age)

    sex_factor = 0.932 if sex_n == "female" else 1.0

    y = Scys / 0.8
    return float(
        133.0
        * (min(y, 1.0) ** -0.499)
        * (max(y, 1.0) ** -1.328)
        * (0.996 ** Age)
        * sex_factor
    )


def egfr_cr_cys(creatinine, cystatin_c, age, sex: Sex) -> float:
    sex_n = _norm_sex(sex)
    Scr = float(creatinine)
    Scys = float(cystatin_c)
    Age = float(age)

    k = 0.7 if sex_n == "female" else 0.9
    a = -0.219 if sex_n == "female" else -0.144
    sex_factor = 0.963 if sex_n == "female" else 1.0

    x = Scr / k
    y = Scys / 0.8

    return float(
        135.0
        * (min(x, 1.0) ** a)
        * (max(x, 1.0) ** -0.544)
        * (min(y, 1.0) ** -0.323)
        * (max(y, 1.0) ** -0.778)
        * (0.9961 ** Age)
        * sex_factor
    )


# -------------------------
# DataFrame API
# -------------------------
def add_derived_features(
    df: pd.DataFrame,
    *,
    age_col: str = "age",
    sex_col: str = "sex",
    creatinine_col: str = "serum_creatinine",
    cystatin_col: str = "cystatin_c",
    urea_col: str = "blood_urea",
) -> pd.DataFrame:
    """
    Adds:
      - egfr_cr, egfr_cys, egfr_cr_cys
      - egfr_discordance = egfr_cys - egfr_cr
      - urea_creatinine_ratio (since CKD.csv has blood_urea not BUN)
    Behavior:
      - If sex column missing -> fallback to "male" for all rows.
      - If cystatin_c missing -> cys/combined/discordance stay NaN.
    """
    out = df.copy()

    # Always create columns so downstream code can rely on them existing
    for col in ["egfr_cr", "egfr_cys", "egfr_cr_cys", "egfr_discordance", "urea_creatinine_ratio"]:
        if col not in out.columns:
            out[col] = np.nan

    # Ratio (doesn't need sex)
    if urea_col in out.columns and creatinine_col in out.columns:
        out["urea_creatinine_ratio"] = _safe_div(out[urea_col], out[creatinine_col])

    # Resolve sex column (fallback to "male" if no sex/gender column exists)
    sex_col_candidates = []
    if sex_col:
        sex_col_candidates.append(sex_col)
    for alt in ["sex", "gender"]:
        if alt not in sex_col_candidates:
            sex_col_candidates.append(alt)

    resolved_sex_col = next((c for c in sex_col_candidates if c in out.columns), None)
    if resolved_sex_col is None:
        out["sex"] = "male"
        resolved_sex_col = "sex"

    # Need age+sex+creatinine to compute egfr_cr
    if not all(c in out.columns for c in [age_col, resolved_sex_col, creatinine_col]):
        return out

    # compute row-wise (sex varies per row)
    def _row_egfr_cr(r):
        if pd.isna(r[age_col]) or pd.isna(r[resolved_sex_col]) or pd.isna(r[creatinine_col]):
            return np.nan
        try:
            return egfr_cr(r[creatinine_col], r[age_col], r[resolved_sex_col])
        except Exception:
            return np.nan

    out["egfr_cr"] = out.apply(_row_egfr_cr, axis=1)

    # Need cystatin for cys/combined/discordance
    if cystatin_col in out.columns:
        def _row_egfr_cys(r):
            if pd.isna(r[age_col]) or pd.isna(r[resolved_sex_col]) or pd.isna(r[cystatin_col]):
                return np.nan
            try:
                return egfr_cys(r[cystatin_col], r[age_col], r[resolved_sex_col])
            except Exception:
                return np.nan

        def _row_egfr_cr_cys(r):
            if (pd.isna(r[age_col]) or pd.isna(r[resolved_sex_col]) or
                pd.isna(r[creatinine_col]) or pd.isna(r[cystatin_col])):
                return np.nan
            try:
                return egfr_cr_cys(
                    r[creatinine_col],
                    r[cystatin_col],
                    r[age_col],
                    r[resolved_sex_col],
                )
            except Exception:
                return np.nan

        out["egfr_cys"] = out.apply(_row_egfr_cys, axis=1)
        out["egfr_cr_cys"] = out.apply(_row_egfr_cr_cys, axis=1)
        out["egfr_discordance"] = out["egfr_cys"] - out["egfr_cr"]

    return out
