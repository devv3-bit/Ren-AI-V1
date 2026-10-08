"""
Step 9: the single held-out evaluation on the 2017-Mar 2020 cycle (P).

Everything evaluated here was frozen by scripts/nhanes_05_train_cv.py
(configurations, fitted estimators, Youden thresholds from train OOF).
This script only loads models and writes metrics; it never refits,
re-tunes or re-selects anything.

Writes
  reports/nhanes/test_metrics.json
  reports/nhanes/test_predictions.csv        SEQN, label, subgroup, weight, prob per model
  reports/nhanes/permutation_importance.csv  final model on the test set (interpretation only)
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.nhanes.constants import COHORT_PATH, EARLY_SUBGROUP_COL, REPORTS_DIR, SEED, TARGET_COL  # noqa: E402
from src.nhanes.evaluate import calibration_table, evaluate_predictions  # noqa: E402
from src.nhanes.features import build_X, split_train_test  # noqa: E402
from src.nhanes.load import load_table  # noqa: E402


def _fmt_auc(block: dict) -> str:
    return f"{block['auc']:.3f} [{block['ci_low']:.3f}, {block['ci_high']:.3f}]"


def main() -> None:
    config = json.loads((REPORTS_DIR / "frozen_config.json").read_text())
    cohort = load_table(COHORT_PATH)
    _, test = split_train_test(cohort)
    y = test[TARGET_COL].to_numpy().astype(int)
    early = test[EARLY_SUBGROUP_COL].to_numpy().astype(bool)
    weights = test["mec_weight"].to_numpy(dtype=float)

    preds = test[["SEQN", "cycle", TARGET_COL, EARLY_SUBGROUP_COL, "mec_weight"]].copy()
    metrics = {
        "evaluated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "frozen_config_created_utc": config["created_utc"],
        "test_cycle": "P (2017-Mar 2020, pre-pandemic)",
        "n_test": int(len(y)),
        "n_ckd": int(y.sum()),
        "prevalence": float(y.mean()),
        "n_early_subgroup": int(early.sum()),
        "n_ckd_early_subgroup": int(y[early].sum()),
        "prevalence_early_subgroup": float(y[early].mean()),
        "final_model": config["final_model"],
        "bootstrap": "1000 resamples, seed 42, percentile 95% CI",
        "models": {},
    }

    for name, spec in config["models"].items():
        est = joblib.load(REPO / spec["estimator_path"])
        X_test = build_X(test, spec["features"], log_cols=spec["log_features"])
        prob = est.predict_proba(X_test)[:, 1]
        preds[f"prob_{name}"] = prob
        m = evaluate_predictions(
            y, prob, spec["threshold"]["threshold"],
            weights=weights, subgroup_mask=early, reference_prevalence=spec["train_prevalence"],
        )
        m["calibration"] = calibration_table(y, prob).to_dict(orient="records")
        m["cv_auc_mean_train"] = spec["cv_auc_mean"]
        m["family"], m["params"], m["n_features"] = spec["family"], spec["params"], spec["n_features"]
        metrics["models"][name] = m

    final = config["final_model"]
    spec = config["models"][final]
    est = joblib.load(REPO / spec["estimator_path"])
    X_test = build_X(test, spec["features"], log_cols=spec["log_features"])
    pi = permutation_importance(est, X_test, y, scoring="roc_auc", n_repeats=5, random_state=SEED, n_jobs=1)
    pi_df = (
        pd.DataFrame({"feature": X_test.columns, "importance_mean": pi.importances_mean, "importance_std": pi.importances_std})
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    pi_df.to_csv(REPORTS_DIR / "permutation_importance.csv", index=False)

    (REPORTS_DIR / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    preds.to_csv(REPORTS_DIR / "test_predictions.csv", index=False)

    print(f"TEST (held out, touched once): n = {len(y):,}, CKD = {int(y.sum()):,} ({y.mean():.1%}); "
          f"early subgroup n = {int(early.sum()):,}, CKD = {int(y[early].sum()):,} ({y[early].mean():.1%})")
    print(f"final model (frozen before this run): {final}\n")
    rows = []
    for name, m in metrics["models"].items():
        t = m["at_train_threshold"]
        rows.append({
            "model": name + (" (FINAL)" if name == final else ""),
            "train CV AUC": f"{m['cv_auc_mean_train']:.3f}",
            "test AUC [95% CI]": _fmt_auc(m["roc_auc"]),
            "early-subgroup AUC [95% CI]": _fmt_auc(m["early_subgroup"]["roc_auc"]),
            "thr": f"{t['threshold']:.3f}",
            "sens": f"{t['sensitivity']:.3f}", "spec": f"{t['specificity']:.3f}",
            "PPV": f"{t['ppv']:.3f}", "NPV": f"{t['npv']:.3f}",
            "Brier": f"{m['brier']:.4f}", "weighted AUC": f"{m['survey_weighted_auc']:.3f}",
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\npermutation importance ({final}, test set, top 10):")
    print(pi_df.head(10).to_string(index=False))

    with (REPORTS_DIR / "run_log.md").open("a") as fh:
        fh.write(f"## Test evaluation {metrics['evaluated_utc']} - scripts/nhanes_06_evaluate_test.py\n\n"
                 f"Single pass on the 2017-2020 cycle (n = {len(y):,}) with the models frozen at "
                 f"{config['created_utc']}. Results in reports/nhanes/test_metrics.json. "
                 f"No model, feature, label, cohort or split decision was changed after this point.\n\n")
    print("\nwritten: test_metrics.json, test_predictions.csv, permutation_importance.csv")


if __name__ == "__main__":
    main()
