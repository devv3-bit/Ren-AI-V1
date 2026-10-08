# src/nhanes/harmonize.py
"""
Map the merged NHANES table onto the repo's canonical internal schema.

Rules (all declared in src/nhanes/constants.py):
  * direct renames for labs / demographics / BMI / urine variables
  * sex 1/2 -> "male"/"female" (+ numeric sex_male for modelling)
  * pregnancy 1 -> 1, 2 -> 0, 3 (cannot ascertain) -> NaN
  * diabetes 1 -> 1, 2 -> 0, 3 (borderline) -> 0, 7/9 -> NaN
  * 2005-06 serum creatinine recalibrated: -0.016 + 0.978 * LBXSCR
  * white cell count x1000 (NHANES reports 1000 cells/uL; UCI uses cells/uL)
  * blood pressure = mean of available readings (0 treated as missing)
  * ACR = URDACT, else URXUMA / URXUCR * 100 (mg/g)
  * MEC weight = WTMEC2YR (2-year cycles) or WTMECPRP (2017-2020)
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.data.validate_labs import validate_labs
from src.nhanes.constants import (
    ACR_FROM_COMPONENTS_FACTOR,
    BP_READING_COLS,
    CREATININE_RECAL,
    CYCLES,
    DIABETES_MAP,
    DIRECT_MAP,
    PREGNANCY_MAP,
    SEX_MALE_MAP,
    SEX_MAP,
    WBC_SCALE,
)


def _numeric(df: pd.DataFrame, col: str) -> pd.Series:
    """Numeric view of a column, or an all-NaN series if the column is absent."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def recalibrate_creatinine(scr: pd.Series, cycle: pd.Series) -> pd.Series:
    """Apply the CDC Deming-regression correction to the 2005-06 cycle only."""
    target = cycle.eq(CREATININE_RECAL["cycle"])
    corrected = CREATININE_RECAL["intercept"] + CREATININE_RECAL["slope"] * scr
    return scr.where(~target, corrected)


def _mean_readings(df: pd.DataFrame, cols, zero_is_missing: bool) -> pd.Series:
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.Series(np.nan, index=df.index, dtype=float)
    vals = df[present].apply(pd.to_numeric, errors="coerce")
    if zero_is_missing:
        vals = vals.mask(vals <= 0)
    return vals.mean(axis=1, skipna=True)


def mean_blood_pressure(df: pd.DataFrame, kind: str) -> pd.Series:
    """Mean of the available readings; P cycle uses the oscillometric columns."""
    is_p = df["cycle"].astype(str).eq("P")
    default = _mean_readings(df, BP_READING_COLS["default"][kind], zero_is_missing=True)
    pcycle = _mean_readings(df, BP_READING_COLS["P"][kind], zero_is_missing=True)
    return pcycle.where(is_p, default)


def albumin_creatinine_ratio(urdact: pd.Series, uma: pd.Series, ucr: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """ACR in mg/g: use URDACT when reported, otherwise compute from components."""
    computed = uma / ucr.replace({0: np.nan}) * ACR_FROM_COMPONENTS_FACTOR
    acr = urdact.combine_first(computed)
    source = pd.Series(
        np.select([urdact.notna(), computed.notna()], ["URDACT", "computed"], default=None),
        index=urdact.index,
        dtype="object",
    )
    return acr, source


def mec_weight(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for cycle, meta in CYCLES.items():
        mask = df["cycle"].astype(str).eq(cycle)
        out = out.where(~mask, _numeric(df, str(meta["weight_col"])))
    return out


def harmonize(merged: pd.DataFrame) -> pd.DataFrame:
    """Return a new frame in the canonical schema (one row per participant)."""
    if "cycle" not in merged.columns or "SEQN" not in merged.columns:
        raise ValueError("merged table must contain SEQN and cycle")

    cols: Dict[str, pd.Series] = {
        "SEQN": merged["SEQN"].astype("int64"),
        "cycle": merged["cycle"].astype(str),
    }
    for raw, canon in DIRECT_MAP.items():
        cols[canon] = _numeric(merged, raw)
    base = pd.DataFrame(cols, index=merged.index)

    sex_code = _numeric(merged, "RIAGENDR")
    acr, acr_source = albumin_creatinine_ratio(
        base["acr"], base["urine_albumin"], base["urine_creatinine"]
    )
    return base.assign(
        sex=sex_code.map(SEX_MAP),
        sex_male=sex_code.map(SEX_MALE_MAP),
        pregnant=_numeric(merged, "RIDEXPRG").map(PREGNANCY_MAP),
        diabetes_mellitus=_numeric(merged, "DIQ010").map(DIABETES_MAP),
        serum_creatinine_raw=base["serum_creatinine"],
        serum_creatinine=recalibrate_creatinine(base["serum_creatinine"], base["cycle"]),
        white_blood_cell_count=base["white_blood_cell_count"] * WBC_SCALE,
        sbp=mean_blood_pressure(merged, "sbp"),
        blood_pressure=mean_blood_pressure(merged, "dbp"),
        acr=acr,
        acr_source=acr_source,
        mec_weight=mec_weight(merged),
    )


# ----------------------------------------------------------------------------
# Biological plausibility (reuses src/data/validate_labs.py + VALIDATION_RULES)
# ----------------------------------------------------------------------------
def plausibility_report(h: pd.DataFrame) -> pd.DataFrame:
    """Count hard-range violations and soft warnings per canonical column."""
    report = validate_labs(h)
    hard = Counter(issue.col for issue in report["hard_errors"])
    soft = Counter(issue.col for issue in report["soft_warnings"])
    from src.config import VALIDATION_RULES

    rows = []
    for col, rule in VALIDATION_RULES.items():
        if col not in h.columns:
            rows.append({"column": col, "present": False, "n_checked": 0,
                         "hard_range": rule.get("hard"), "hard_violations": 0, "soft_warnings": 0})
            continue
        rows.append(
            {
                "column": col,
                "present": True,
                "n_checked": int(pd.to_numeric(h[col], errors="coerce").notna().sum()),
                "hard_range": rule.get("hard"),
                "hard_violations": int(hard.get(col, 0)),
                "soft_warnings": int(soft.get(col, 0)),
            }
        )
    return pd.DataFrame(rows)
