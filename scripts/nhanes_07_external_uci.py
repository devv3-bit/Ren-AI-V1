"""
Step 10: external validation on the UCI CKD hospital patients (399 usable rows).

The harmonised Model A frozen in Step 8 (trained and tuned on NHANES
2005-2016 only, 14 features present in both datasets) is applied unchanged.
UCI has no sex column: eGFR is computed as male (the repo's v1 fallback) for
the primary result and as female for a sensitivity analysis.

Writes reports/nhanes/external_metrics.json and reports/nhanes/uci_predictions.csv
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.nhanes.constants import REPORTS_DIR  # noqa: E402
from src.nhanes.evaluate import confusion_at, evaluate_predictions  # noqa: E402
from src.nhanes.external import SEX_ASSUMPTIONS, feature_availability, load_uci_harmonized, uci_target  # noqa: E402
from src.nhanes.features import build_X  # noqa: E402

MODEL_NAME = "model_A_harmonized"


def main() -> None:
    config = json.loads((REPORTS_DIR / "frozen_config.json").read_text())
    spec = config["models"][MODEL_NAME]
    est = joblib.load(REPO / spec["estimator_path"])
    threshold = spec["threshold"]["threshold"]

    metrics = {
        "evaluated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "model": MODEL_NAME,
        "family": spec["family"],
        "params": spec["params"],
        "features": spec["features"],
        "threshold_from_nhanes_train_youden": threshold,
        "nhanes_train_cv_auc": spec["cv_auc_mean"],
        "by_sex_assumption": {},
    }
    preds = None
    for sex in SEX_ASSUMPTIONS:
        uci = load_uci_harmonized(sex)
        y = uci_target(uci)
        X = build_X(uci, spec["features"], log_cols=spec["log_features"])
        prob = est.predict_proba(X)[:, 1]
        m = evaluate_predictions(y, prob, threshold, reference_prevalence=spec["train_prevalence"])
        m["at_threshold_0.5"] = confusion_at(y, prob, 0.5)
        m["feature_availability"] = feature_availability(uci, spec["features"]).to_dict(orient="records")
        metrics["by_sex_assumption"][sex] = m
        if preds is None:
            preds = pd.DataFrame({"row": range(len(y)), "ckd": y})
            metrics["n_uci"] = int(len(y))
            metrics["uci_prevalence"] = float(y.mean())
        preds[f"prob_egfr_{sex}"] = prob

    (REPORTS_DIR / "external_metrics.json").write_text(json.dumps(metrics, indent=2))
    preds.to_csv(REPORTS_DIR / "uci_predictions.csv", index=False)

    print(f"UCI hospital patients: n = {metrics['n_uci']}, CKD prevalence = {metrics['uci_prevalence']:.1%}")
    print(f"model: {MODEL_NAME} {spec['params']} on {len(spec['features'])} shared features; "
          f"threshold {threshold:.3f} (Youden on NHANES train OOF)\n")
    for sex, m in metrics["by_sex_assumption"].items():
        a, t, t5 = m["roc_auc"], m["at_train_threshold"], m["at_threshold_0.5"]
        print(f"eGFR as {sex:<6s}: AUC {a['auc']:.3f} [{a['ci_low']:.3f}, {a['ci_high']:.3f}]  "
              f"sens {t['sensitivity']:.3f} spec {t['specificity']:.3f} PPV {t['ppv']:.3f} NPV {t['npv']:.3f} "
              f"(at 0.5: sens {t5['sensitivity']:.3f} spec {t5['specificity']:.3f})  Brier {m['brier']:.3f}")
    avail = pd.DataFrame(metrics["by_sex_assumption"]["male"]["feature_availability"])
    print("\nUCI feature availability:")
    print(avail.to_string(index=False))

    with (REPORTS_DIR / "run_log.md").open("a") as fh:
        fh.write(f"## External validation {metrics['evaluated_utc']} - scripts/nhanes_07_external_uci.py\n\n"
                 f"Frozen {MODEL_NAME} applied unchanged to the {metrics['n_uci']} UCI patients "
                 f"(eGFR as male primary, female sensitivity). Results in reports/nhanes/external_metrics.json.\n\n")
    print("\nwritten: external_metrics.json, uci_predictions.csv")


if __name__ == "__main__":
    main()
