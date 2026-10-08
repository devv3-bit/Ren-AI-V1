"""
POST-HOC diagnostics for the external validation (exploratory, NOT pre-registered).

Nothing here is retrained or re-selected. The frozen harmonised Model A is
applied to subsets of the UCI data to explain WHY the pre-registered
external AUC (reports/nhanes/external_metrics.json) is what it is. The
pre-registered number remains the result; these numbers are context.

Writes reports/nhanes/external_diagnostics.json
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.nhanes.constants import COHORT_PATH, REPORTS_DIR  # noqa: E402
from src.nhanes.evaluate import bootstrap_auc_ci  # noqa: E402
from src.nhanes.external import load_uci_harmonized, uci_target  # noqa: E402
from src.nhanes.features import build_X, split_train_test  # noqa: E402
from src.nhanes.load import load_table  # noqa: E402

SINGLE_FEATURES = [("egfr_cr", -1), ("serum_creatinine", 1), ("bun", 1), ("hemoglobin", -1),
                   ("packed_cell_volume", -1), ("blood_glucose_random", 1), ("bun_creatinine_ratio", 1)]


def main() -> None:
    config = json.loads((REPORTS_DIR / "frozen_config.json").read_text())
    spec = config["models"]["model_A_harmonized"]
    est = joblib.load(REPO / spec["estimator_path"])
    features = spec["features"]

    uci = load_uci_harmonized("male")
    y = uci_target(uci)
    X = build_X(uci, features, log_cols=spec["log_features"])
    prob = est.predict_proba(X)[:, 1]

    # 1) single-feature ceilings on UCI (no model)
    single = []
    for f, sign in SINGLE_FEATURES:
        v = pd.to_numeric(uci[f], errors="coerce")
        m = v.notna().to_numpy()
        single.append({"feature": f, "sign": sign, "n": int(m.sum()),
                       "auc": float(roc_auc_score(y[m], sign * v[m]))})

    # 2) frozen model on UCI subsets defined by data completeness
    complete = X.notna().all(axis=1).to_numpy()
    k_na_present = X[["potassium", "sodium"]].notna().all(axis=1).to_numpy()
    subsets = {
        "all_rows": np.ones(len(y), dtype=bool),
        "complete_cases_all_14_features": complete,
        "potassium_and_sodium_observed": k_na_present,
        "potassium_or_sodium_missing": ~k_na_present,
    }
    subset_results = {}
    for name, m in subsets.items():
        subset_results[name] = {
            **bootstrap_auc_ci(y[m], prob[m]),
            "prevalence": float(y[m].mean()) if m.any() else float("nan"),
            "median_prob_ckd": float(np.median(prob[m & (y == 1)])) if (m & (y == 1)).any() else float("nan"),
            "median_prob_not_ckd": float(np.median(prob[m & (y == 0)])) if (m & (y == 0)).any() else float("nan"),
        }

    # 3) missingness by class in UCI
    miss_by_class = {
        f: {"missing_frac_ckd": float(X.loc[y == 1, f].isna().mean()),
            "missing_frac_not_ckd": float(X.loc[y == 0, f].isna().mean())}
        for f in features
    }

    # 4) why: standardised missing indicators learned on NHANES train
    cohort = load_table(COHORT_PATH)
    train, _ = split_train_test(cohort)
    X_train = build_X(train, features, log_cols=spec["log_features"])
    imputer = est.named_steps["preprocess"].named_steps["imputer"]
    scaler = est.named_steps["preprocess"].named_steps["scaler"]
    names = list(imputer.get_feature_names_out(input_features=features))
    coef = est.named_steps["clf"].coef_.ravel()
    indicator_effects = []
    for i, n in enumerate(names):
        if not n.startswith("missingindicator_"):
            continue
        f = n[len("missingindicator_"):]
        p = float(X_train[f].isna().mean())
        z_missing = float((1.0 - scaler.mean_[i]) / scaler.scale_[i])
        indicator_effects.append({
            "feature": f,
            "nhanes_train_missing_rate": p,
            "z_value_when_missing": z_missing,
            "coef_per_sd": float(coef[i]),
            "log_odds_shift_when_missing": float(coef[i] * z_missing),
            "uci_missing_rate": float(X[f].isna().mean()),
        })

    # 5) mean contribution to the log-odds by UCI class
    Z = est.named_steps["preprocess"].transform(X)
    contrib = pd.DataFrame(Z * coef, columns=names).assign(y=y).groupby("y").mean().T
    contrib_records = [{"term": idx, "mean_not_ckd": float(r[0]), "mean_ckd": float(r[1]),
                        "diff_ckd_minus_not": float(r[1] - r[0])} for idx, r in contrib.iterrows()]

    out = {
        "evaluated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "status": "POST-HOC, exploratory. Not pre-registered. The pre-registered external result is in external_metrics.json.",
        "model": "model_A_harmonized (frozen; eGFR as male)",
        "single_feature_auc_uci": single,
        "frozen_model_on_uci_subsets": subset_results,
        "uci_missingness_by_class": miss_by_class,
        "standardised_missing_indicator_effects": indicator_effects,
        "mean_log_odds_contribution_by_class": contrib_records,
    }
    (REPORTS_DIR / "external_diagnostics.json").write_text(json.dumps(out, indent=2))

    print("POST-HOC UCI diagnostics (exploratory)")
    print("\nsingle-feature AUCs on UCI (no model):")
    for s in single:
        print(f"  {s['feature']:<22s} sign {s['sign']:+d}  n={s['n']:3d}  AUC {s['auc']:.3f}")
    print("\nfrozen harmonised Model A on UCI subsets:")
    for k, v in subset_results.items():
        print(f"  {k:<34s} n={v['n']:3d} (CKD {v['prevalence']:.1%})  AUC {v['auc']:.3f} [{v['ci_low']:.3f}, {v['ci_high']:.3f}]"
              f"  median p(CKD|CKD)={v['median_prob_ckd']:.3f}  p(CKD|not)={v['median_prob_not_ckd']:.3f}")
    print("\nstandardised missing-indicator effects (NHANES train -> UCI):")
    print(pd.DataFrame(indicator_effects).round(4).to_string(index=False))
    with (REPORTS_DIR / "run_log.md").open("a") as fh:
        fh.write(f"## Post-hoc UCI diagnostics {out['evaluated_utc']} - scripts/nhanes_09_uci_diagnostics.py\n\n"
                 "Exploratory, not pre-registered, nothing retrained: the frozen harmonised Model A applied to UCI "
                 "subsets to explain the external result. See reports/nhanes/external_diagnostics.json.\n\n")


if __name__ == "__main__":
    main()
