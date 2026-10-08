import numpy as np
import pandas as pd
import pytest

from src.nhanes.evaluate import bootstrap_auc_ci, calibration_table, confusion_at, evaluate_predictions
from src.nhanes.models import (
    best_result,
    build_estimator,
    cv_auc,
    make_cv,
    oof_probabilities,
    tune_logistic,
    youden_threshold,
)


def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    X.loc[rng.random(n) < 0.1, "b"] = np.nan
    y = ((X["a"] + rng.normal(scale=0.7, size=n)) > 0).astype(int).to_numpy()
    return X, y


def test_youden_threshold_on_separable_scores():
    y = np.array([0, 0, 0, 1, 1, 1])
    prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    t = youden_threshold(y, prob)
    assert t["sensitivity"] == 1.0 and t["specificity"] == 1.0 and t["youden_j"] == 1.0
    assert 0.3 < t["threshold"] <= 0.7


def test_confusion_at_counts():
    y = np.array([1, 1, 0, 0, 1])
    prob = np.array([0.9, 0.4, 0.6, 0.1, 0.5])
    c = confusion_at(y, prob, 0.5)
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (2, 1, 1, 1)
    assert c["sensitivity"] == pytest.approx(2 / 3)
    assert c["specificity"] == pytest.approx(1 / 2)
    assert c["ppv"] == pytest.approx(2 / 3)
    assert c["npv"] == pytest.approx(1 / 2)


def test_bootstrap_ci_is_deterministic_and_brackets_point_estimate():
    X, y = _toy()
    prob = 1 / (1 + np.exp(-X["a"].to_numpy()))
    a = bootstrap_auc_ci(y, prob, n_boot=200, seed=42)
    b = bootstrap_auc_ci(y, prob, n_boot=200, seed=42)
    assert a == b
    assert a["ci_low"] <= a["auc"] <= a["ci_high"]
    assert a["n_boot_valid"] == 200


def test_cv_tuning_and_selection_are_reproducible():
    X, y = _toy()
    cv = make_cv()
    r1 = tune_logistic("demo", X, y, cv, c_grid=[0.1, 1.0])
    r2 = tune_logistic("demo", X, y, cv, c_grid=[0.1, 1.0])
    assert [r.mean_auc for r in r1] == [r.mean_auc for r in r2]
    assert all(0.5 < r.mean_auc <= 1.0 and len(r.fold_aucs) == 5 for r in r1)
    best = best_result(r1, "demo")
    assert best.mean_auc == max(r.mean_auc for r in r1)
    with pytest.raises(ValueError):
        best_result(r1, "missing")


def test_hgb_config_runs_and_handles_nan():
    X, y = _toy()
    r = cv_auc("hgb", "hgb", {"learning_rate": 0.1, "max_depth": 3, "max_iter": 30}, X, y, make_cv())
    assert 0.5 < r.mean_auc <= 1.0
    with pytest.raises(ValueError):
        build_estimator("svm", {})


def test_evaluate_predictions_shapes():
    X, y = _toy()
    oof = oof_probabilities(build_estimator("logistic", {"C": 1.0}), X, y, make_cv())
    thr = youden_threshold(y, oof)["threshold"]
    m = evaluate_predictions(y, oof, thr, weights=np.ones(len(y)), subgroup_mask=X["a"] > 0,
                             reference_prevalence=float(y.mean()), n_boot=50)
    assert set(m) >= {"roc_auc", "at_train_threshold", "brier", "early_subgroup", "survey_weighted_auc"}
    assert m["survey_weighted_auc"] == pytest.approx(m["roc_auc"]["auc"])
    assert 0 <= m["brier"] <= m["brier_reference_constant"] + 0.05
    cal = calibration_table(y, oof, n_bins=5)
    assert len(cal) == 5 and cal["n"].sum() == len(y)
