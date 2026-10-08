# src/nhanes/evaluate.py
"""
Metrics for the single held-out evaluation (Step 9) and the external
validation (Step 10). Nothing here fits a model.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.nhanes.constants import N_BOOTSTRAP, SEED


def bootstrap_auc_ci(y, prob, n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> Dict[str, float]:
    """ROC-AUC with a percentile bootstrap 95% CI (resampling rows with replacement)."""
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, dtype=float)
    n = len(y)
    if n == 0 or y.min() == y.max():
        return {"auc": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "n": int(n), "n_pos": int(y.sum()), "n_boot_valid": 0}
    rng = np.random.default_rng(seed)
    aucs = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() != yb.max():
            aucs[b] = roc_auc_score(yb, prob[idx])
    return {
        "auc": float(roc_auc_score(y, prob)),
        "ci_low": float(np.nanpercentile(aucs, 2.5)),
        "ci_high": float(np.nanpercentile(aucs, 97.5)),
        "n": int(n),
        "n_pos": int(y.sum()),
        "n_boot_valid": int(np.isfinite(aucs).sum()),
    }


def _ratio(num: int, den: int) -> float:
    return float(num / den) if den else float("nan")


def confusion_at(y, prob, threshold: float) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(prob, dtype=float) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return {
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "sensitivity": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
        "ppv": _ratio(tp, tp + fp),
        "npv": _ratio(tn, tn + fn),
        "accuracy": _ratio(tp + tn, len(y)),
        "flagged_fraction": _ratio(tp + fp, len(y)),
    }


def brier(y, prob) -> float:
    return float(brier_score_loss(np.asarray(y).astype(int), np.asarray(prob, dtype=float)))


def calibration_table(y, prob, n_bins: int = 10) -> pd.DataFrame:
    """Quantile-binned reliability table: mean predicted vs observed fraction."""
    df = pd.DataFrame({"y": np.asarray(y).astype(int), "prob": np.asarray(prob, dtype=float)})
    df = df.assign(bin=pd.qcut(df["prob"].rank(method="first"), q=n_bins, labels=False))
    out = df.groupby("bin").agg(mean_pred=("prob", "mean"), frac_pos=("y", "mean"), n=("y", "size")).reset_index()
    return out


def weighted_auc(y, prob, weights) -> float:
    w = pd.to_numeric(pd.Series(weights), errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(w) & (w > 0)
    y = np.asarray(y).astype(int)[keep]
    if len(y) == 0 or y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, np.asarray(prob, dtype=float)[keep], sample_weight=w[keep]))


def evaluate_predictions(
    y,
    prob,
    threshold: float,
    *,
    weights=None,
    subgroup_mask=None,
    reference_prevalence: Optional[float] = None,
    n_boot: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> Dict[str, object]:
    """All Step 9 metrics for one model's predictions on one evaluation set."""
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, dtype=float)
    out: Dict[str, object] = {
        "roc_auc": bootstrap_auc_ci(y, prob, n_boot=n_boot, seed=seed),
        "at_train_threshold": confusion_at(y, prob, threshold),
        "brier": brier(y, prob),
        "prevalence": float(y.mean()),
    }
    if reference_prevalence is not None:
        out["brier_reference_constant"] = float(np.mean((y - reference_prevalence) ** 2))
    if weights is not None:
        out["survey_weighted_auc"] = weighted_auc(y, prob, weights)
    if subgroup_mask is not None:
        m = np.asarray(subgroup_mask).astype(bool)
        out["early_subgroup"] = {
            "roc_auc": bootstrap_auc_ci(y[m], prob[m], n_boot=n_boot, seed=seed),
            "at_train_threshold": confusion_at(y[m], prob[m], threshold),
            "prevalence": float(y[m].mean()) if m.any() else float("nan"),
        }
    return out
