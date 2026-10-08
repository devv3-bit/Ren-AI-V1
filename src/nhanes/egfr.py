# src/nhanes/egfr.py
"""
Vectorised CKD-EPI 2021 (race-free) creatinine eGFR.

The scalar reference implementation is src/features/derive.py::egfr_cr. This
module re-implements the same equation with numpy so that it can be applied
to ~70k NHANES rows at once. tests/test_nhanes_egfr.py checks that the two
agree to floating-point precision.
"""
from __future__ import annotations

from typing import Iterable, Union

import numpy as np
import pandas as pd

_FEMALE = {"f", "female"}
_MALE = {"m", "male"}

# CKD-EPI 2021 constants (identical to src/features/derive.py)
_K = {"female": 0.7, "male": 0.9}
_ALPHA = {"female": -0.241, "male": -0.302}
_SEX_FACTOR = {"female": 1.012, "male": 1.0}
_BETA = -1.200
_AGE_FACTOR = 0.9938
_SCALE = 142.0


def _as_float_array(values: Union[Iterable, pd.Series, np.ndarray]) -> np.ndarray:
    return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)


def _sex_masks(sex: Union[Iterable, pd.Series]) -> tuple[np.ndarray, np.ndarray]:
    s = pd.Series(sex).astype("string").str.strip().str.lower()
    is_female = s.isin(list(_FEMALE)).fillna(False).to_numpy(dtype=bool)
    is_male = s.isin(list(_MALE)).fillna(False).to_numpy(dtype=bool)
    return is_female, is_male


def egfr_cr_vectorized(creatinine, age, sex) -> np.ndarray:
    """
    CKD-EPI 2021 creatinine eGFR for arrays.

    Parameters
    ----------
    creatinine : serum creatinine, mg/dL
    age        : years
    sex        : "male"/"female" (or "m"/"f"), one per row

    Returns NaN where creatinine, age or sex is missing/invalid, or where
    creatinine <= 0 (the scalar version raises in that case).
    """
    scr = _as_float_array(creatinine)
    yrs = _as_float_array(age)
    is_female, is_male = _sex_masks(sex)

    if not (len(scr) == len(yrs) == len(is_female)):
        raise ValueError("creatinine, age and sex must have the same length")

    k = np.where(is_female, _K["female"], _K["male"])
    alpha = np.where(is_female, _ALPHA["female"], _ALPHA["male"])
    sex_factor = np.where(is_female, _SEX_FACTOR["female"], _SEX_FACTOR["male"])

    valid = np.isfinite(scr) & np.isfinite(yrs) & (is_female | is_male) & (scr > 0)

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        x = scr / k
        out = (
            _SCALE
            * np.minimum(x, 1.0) ** alpha
            * np.maximum(x, 1.0) ** _BETA
            * (_AGE_FACTOR ** yrs)
            * sex_factor
        )
    return np.where(valid, out, np.nan)
