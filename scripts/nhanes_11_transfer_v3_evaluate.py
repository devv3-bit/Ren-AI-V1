"""
v3 transfer model, part 2: the disclosed SECOND look at the 399 UCI patients.

Applies the models frozen by scripts/nhanes_10_transfer_v3_train.py unchanged.
Also scores them on the NHANES 2017-2020 cycle for context (a second look at
that cycle too; no NHANES-level decision depends on it).

Writes reports/nhanes/external_metrics_v3.json, uci_predictions_v3.csv, run_log.md (append)
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.nhanes.constants import COHORT_PATH, EARLY_SUBGROUP_COL, REPORTS_DIR, TARGET_COL  # noqa: E402
from src.nhanes.evaluate import bootstrap_auc_ci, confusion_at, evaluate_predictions  # noqa: E402
from src.nhanes.external import SEX_ASSUMPTIONS, load_uci_harmonized, uci_target  # noqa: E402
from src.nhanes.features import build_X, split_train_test  # noqa: E402
from src.nhanes.load import load_table  # noqa: E402


def _fmt(block: dict) -> str:
    return f"{block['auc']:.3f} [{block['ci_low']:.3f}, {block['ci_high']:.3f}]"


def _subsets(X: pd.DataFrame) -> dict:
    k_na = X[["potassium", "sodium"]].notna().all(axis=1).to_numpy()
    return {
        "complete_cases_all_14_features": X.notna().all(axis=1).to_numpy(),
        "potassium_and_sodium_observed": k_na,
        "potassium_or_sodium_missing": ~k_na,
    }


def main() -> None:
    config = json.loads((REPORTS_DIR / "frozen_config_v3.json").read_text())
    v2 = json.loads((REPORTS_DIR / "external_metrics.json").read_text())
    cohort = load_table(COHORT_PATH)
    _, test = split_train_test(cohort)
    y_test = test[TARGET_COL].to_numpy().astype(int)
    early = test[EARLY_SUBGROUP_COL].to_numpy().astype(bool)

    out = {
        "evaluated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "status": "SECOND look at UCI (disclosed). v2 first-look result: external_metrics.json.",
        "frozen_config_created_utc": config["created_utc"],
        "final_model": config["final_model"],
        "v2_first_look_auc_male": v2["by_sex_assumption"]["male"]["roc_auc"],
        "models": {},
    }
    preds = None
    for name, spec in config["models"].items():
        est = joblib.load(REPO / spec["estimator_path"])
        thr = spec["threshold"]["threshold"]
        entry = {"role": spec["role"], "family": spec["family"], "params": spec["params"],
                 "nhanes_train_cv_auc": spec["cv_auc_mean"], "threshold_train_youden": thr, "uci": {}}
        for sex in SEX_ASSUMPTIONS:
            uci = load_uci_harmonized(sex)
            y = uci_target(uci)
            X = build_X(uci, spec["features"], log_cols=spec["log_features"])
            prob = est.predict_proba(X)[:, 1]
            m = evaluate_predictions(y, prob, thr, reference_prevalence=spec["train_prevalence"])
            m["at_threshold_0.5"] = confusion_at(y, prob, 0.5)
            if sex == "male":
                m["subsets"] = {k: {**bootstrap_auc_ci(y[mask], prob[mask]), "prevalence": float(y[mask].mean())}
                                for k, mask in _subsets(X).items()}
                if preds is None:
                    preds = pd.DataFrame({"row": range(len(y)), "ckd": y})
                    out["n_uci"], out["uci_prevalence"] = int(len(y)), float(y.mean())
                preds[f"prob_{name}"] = prob
            entry["uci"][sex] = m
        X_t = build_X(test, spec["features"], log_cols=spec["log_features"])
        p_t = est.predict_proba(X_t)[:, 1]
        entry["nhanes_test_context"] = {
            "roc_auc": bootstrap_auc_ci(y_test, p_t),
            "early_subgroup_roc_auc": bootstrap_auc_ci(y_test[early], p_t[early]),
            "at_train_threshold": confusion_at(y_test, p_t, thr),
        }
        out["models"][name] = entry

    (REPORTS_DIR / "external_metrics_v3.json").write_text(json.dumps(out, indent=2))
    preds.to_csv(REPORTS_DIR / "uci_predictions_v3.csv", index=False)

    print(f"UCI second look (disclosed): n = {out['n_uci']}, CKD {out['uci_prevalence']:.1%}; "
          f"v2 first look AUC {out['v2_first_look_auc_male']['auc']:.3f}\n")
    rows = []
    for name, e in out["models"].items():
        um, uf, nt = e["uci"]["male"], e["uci"]["female"], e["nhanes_test_context"]
        t = um["at_train_threshold"]
        rows.append({"model": name + (" (FINAL v3)" if name == out["final_model"] else f" ({e['role']})"),
                     "NHANES CV": f"{e['nhanes_train_cv_auc']:.3f}",
                     "UCI AUC male [CI]": _fmt(um["roc_auc"]), "UCI AUC female": f"{uf['roc_auc']['auc']:.3f}",
                     "sens": f"{t['sensitivity']:.3f}", "spec": f"{t['specificity']:.3f}",
                     "PPV": f"{t['ppv']:.3f}", "NPV": f"{t['npv']:.3f}", "Brier": f"{um['brier']:.3f}",
                     "NHANES test AUC": _fmt(nt["roc_auc"]), "early": f"{nt['early_subgroup_roc_auc']['auc']:.3f}"})
    print(pd.DataFrame(rows).to_string(index=False))
    fin = out["models"][out["final_model"]]["uci"]["male"]["subsets"]
    print(f"\nmechanism check, final v3 model on UCI subsets:")
    for k, v in fin.items():
        print(f"  {k:<34s} n={v['n']:3d} (CKD {v['prevalence']:.1%})  AUC {v['auc']:.3f} [{v['ci_low']:.3f}, {v['ci_high']:.3f}]")
    with (REPORTS_DIR / "run_log.md").open("a") as fh:
        fh.write(f"## v3 external evaluation {out['evaluated_utc']} - scripts/nhanes_11_transfer_v3_evaluate.py\n\n"
                 f"Second, disclosed look at UCI with the models frozen at {config['created_utc']}. "
                 f"Results in external_metrics_v3.json; v2 first-look AUC {out['v2_first_look_auc_male']['auc']:.3f} stays on record.\n\n")
    print("\nwritten: external_metrics_v3.json, uci_predictions_v3.csv")


if __name__ == "__main__":
    main()
