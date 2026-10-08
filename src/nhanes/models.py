# src/nhanes/models.py
"""
Model definitions and cross-validated tuning (Step 8).

Everything here is fitted with 5-fold stratified CV on the TRAIN cycles
only. The test cycle is never passed to this module.

  Model A (primary, interpretable)
      median imputer + missing indicators -> StandardScaler ->
      L2 LogisticRegression(class_weight="balanced"); C tuned over
      LOGISTIC_C_GRID. Same structure as v1 (build_features.make_preprocessor
      + src/models/train_logistic.py).
  Model B (comparison)
      HistGradientBoostingClassifier over a small grid (NaN handled natively).
  Baselines
      eGFR alone; age + sex + diabetes + BP. Both use the Model A pipeline.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from src.features.build_features import make_preprocessor
from src.nhanes.constants import HGB_GRID, LOGISTIC_C_GRID, N_SPLITS, SEED


# ----------------------------------------------------------------------------
# Estimators
# ----------------------------------------------------------------------------
def make_logistic(C: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor(add_missing_indicators=True)),
            ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=SEED)),
        ]
    )


def make_hgb(learning_rate: float, max_depth: int, max_iter: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_depth=max_depth,
        max_iter=max_iter,
        early_stopping=False,
        random_state=SEED,
    )


def build_estimator(family: str, params: Dict[str, object]) -> BaseEstimator:
    if family == "logistic":
        return make_logistic(float(params["C"]))
    if family == "hgb":
        return make_hgb(float(params["learning_rate"]), int(params["max_depth"]), int(params["max_iter"]))
    raise ValueError(f"unknown model family: {family}")


def make_cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)


# ----------------------------------------------------------------------------
# Cross-validation
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class CVResult:
    model_set: str
    family: str
    params: Dict[str, object]
    n_features: int
    fold_aucs: List[float]
    mean_auc: float
    std_auc: float
    seconds: float

    def to_record(self) -> Dict[str, object]:
        rec = asdict(self)
        rec["params"] = ", ".join(f"{k}={v}" for k, v in self.params.items())
        rec["fold_aucs"] = ", ".join(f"{a:.4f}" for a in self.fold_aucs)
        return rec


def cv_auc(
    model_set: str,
    family: str,
    params: Dict[str, object],
    X: pd.DataFrame,
    y: np.ndarray,
    cv: StratifiedKFold,
) -> CVResult:
    """Fold-wise ROC-AUC of one configuration (fit on k-1 folds, scored on the held-out fold)."""
    y = np.asarray(y).astype(int)
    start = time.perf_counter()
    fold_aucs: List[float] = []
    for train_idx, valid_idx in cv.split(X, y):
        est = build_estimator(family, params)
        est.fit(X.iloc[train_idx], y[train_idx])
        prob = est.predict_proba(X.iloc[valid_idx])[:, 1]
        fold_aucs.append(float(roc_auc_score(y[valid_idx], prob)))
    return CVResult(
        model_set=model_set,
        family=family,
        params=dict(params),
        n_features=int(X.shape[1]),
        fold_aucs=fold_aucs,
        mean_auc=float(np.mean(fold_aucs)),
        std_auc=float(np.std(fold_aucs, ddof=1)),
        seconds=float(time.perf_counter() - start),
    )


def tune_logistic(
    model_set: str, X: pd.DataFrame, y: np.ndarray, cv: StratifiedKFold,
    c_grid: Sequence[float] = LOGISTIC_C_GRID,
) -> List[CVResult]:
    return [cv_auc(model_set, "logistic", {"C": float(C)}, X, y, cv) for C in c_grid]


def tune_hgb(
    model_set: str, X: pd.DataFrame, y: np.ndarray, cv: StratifiedKFold,
    grid: Dict[str, List[object]] = HGB_GRID,
) -> List[CVResult]:
    keys = list(grid.keys())
    results = []
    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, values))
        results.append(cv_auc(model_set, "hgb", params, X, y, cv))
    return results


def best_result(results: Sequence[CVResult], model_set: str) -> CVResult:
    """Highest mean CV AUC within a model set; ties resolve to the earlier grid entry."""
    candidates = [r for r in results if r.model_set == model_set]
    if not candidates:
        raise ValueError(f"no CV results for model set {model_set!r}")
    return max(candidates, key=lambda r: (r.mean_auc, -candidates.index(r)))


def results_table(results: Sequence[CVResult]) -> pd.DataFrame:
    return pd.DataFrame([r.to_record() for r in results])


# ----------------------------------------------------------------------------
# Out-of-fold predictions and threshold (chosen on TRAIN CV only)
# ----------------------------------------------------------------------------
def oof_probabilities(estimator: BaseEstimator, X: pd.DataFrame, y: np.ndarray, cv: StratifiedKFold) -> np.ndarray:
    return cross_val_predict(clone(estimator), X, np.asarray(y).astype(int), cv=cv, method="predict_proba")[:, 1]


def youden_threshold(y: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    """Threshold maximising Youden's J = sensitivity + specificity - 1."""
    fpr, tpr, thr = roc_curve(np.asarray(y).astype(int), np.asarray(prob, dtype=float))
    j = tpr - fpr
    i = int(np.argmax(j))
    threshold = float(thr[i]) if np.isfinite(thr[i]) else 1.0
    return {
        "threshold": threshold,
        "sensitivity": float(tpr[i]),
        "specificity": float(1.0 - fpr[i]),
        "youden_j": float(j[i]),
    }
