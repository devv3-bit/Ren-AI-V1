"""
Step 8: tune and FREEZE every model with 5-fold stratified CV on the TRAIN
cycles (2005-2016) only. The 2017-2020 test cycle is never loaded here.

Writes
  reports/nhanes/cv_results.csv            every configuration tried
  reports/nhanes/frozen_config.json        chosen configs + Youden thresholds (train OOF)
  reports/nhanes/model_A_coefficients.csv  odds ratios of the frozen Model A
  reports/nhanes/run_log.md                append-only log of this run
  data/external/nhanes/models/*.joblib     frozen fitted estimators (git-ignored)
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.nhanes.constants import (  # noqa: E402
    COHORT_PATH, DEMOGRAPHIC_FEATURES, EGFR_ONLY_FEATURES, HARMONIZED_FEATURES, HGB_GRID,
    LOGISTIC_C_GRID, LOG_FEATURES, MODELS_DIR, MODEL_FEATURES, N_SPLITS, REPORTS_DIR, SEED,
    TARGET_COL,
)
from src.nhanes.features import build_X, split_train_test  # noqa: E402
from src.nhanes.load import load_table  # noqa: E402
from src.nhanes.models import (  # noqa: E402
    best_result, build_estimator, make_cv, oof_probabilities, results_table, tune_hgb,
    tune_logistic, youden_threshold,
)

FEATURE_SETS = {
    "model_A_full": (MODEL_FEATURES, LOG_FEATURES),
    "model_B_full": (MODEL_FEATURES, LOG_FEATURES),
    "baseline_egfr": (EGFR_ONLY_FEATURES, []),
    "baseline_demographic": (DEMOGRAPHIC_FEATURES, []),
    "model_A_harmonized": (HARMONIZED_FEATURES, LOG_FEATURES),
}
CANDIDATES_FOR_FINAL = ["model_A_full", "model_B_full"]


def _log(msg: str) -> None:
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def model_a_coefficients(est, feature_names) -> pd.DataFrame:
    imputer = est.named_steps["preprocess"].named_steps["imputer"]
    names = imputer.get_feature_names_out(input_features=list(feature_names))
    coef = est.named_steps["clf"].coef_.ravel()
    return (
        pd.DataFrame({"feature": names, "coef_log_odds": coef, "odds_ratio": np.exp(coef)})
        .sort_values("coef_log_odds", key=np.abs, ascending=False)
        .reset_index(drop=True)
    )


def main() -> None:
    cohort = load_table(COHORT_PATH)
    train, _ = split_train_test(cohort)  # the test frame is discarded immediately
    y = train[TARGET_COL].to_numpy().astype(int)
    cv = make_cv()
    _log(f"train N = {len(y):,}, CKD prevalence = {y.mean():.4f}; CV = StratifiedKFold({N_SPLITS}, shuffle, seed {SEED})")

    X = {name: build_X(train, cols, log_cols=logs) for name, (cols, logs) in FEATURE_SETS.items()}

    results = []
    _log(f"Model A grid C={LOGISTIC_C_GRID} on {X['model_A_full'].shape[1]} features")
    results += tune_logistic("model_A_full", X["model_A_full"], y, cv)
    _log(f"Model B grid {HGB_GRID}")
    results += tune_hgb("model_B_full", X["model_B_full"], y, cv)
    _log("baseline: eGFR alone")
    results += tune_logistic("baseline_egfr", X["baseline_egfr"], y, cv)
    _log("baseline: age + sex + diabetes + BP")
    results += tune_logistic("baseline_demographic", X["baseline_demographic"], y, cv)
    _log(f"harmonised Model A (UCI-compatible, {X['model_A_harmonized'].shape[1]} features)")
    results += tune_logistic("model_A_harmonized", X["model_A_harmonized"], y, cv)

    table = results_table(results)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS_DIR / "cv_results.csv", index=False)
    print("\n=== all configurations (5-fold CV ROC-AUC on train) ===")
    print(table[["model_set", "family", "params", "n_features", "mean_auc", "std_auc", "fold_aucs", "seconds"]]
          .to_string(index=False))

    chosen = {name: best_result(results, name) for name in FEATURE_SETS}
    final_name = max(CANDIDATES_FOR_FINAL, key=lambda n: chosen[n].mean_auc)
    print("\n=== chosen per model set ===")
    for name, res in chosen.items():
        print(f"  {name:<22s} {res.family:<9s} {res.params}  CV AUC {res.mean_auc:.4f} +/- {res.std_auc:.4f}")
    print(f"\nFINAL MODEL (highest mean CV AUC among {CANDIDATES_FOR_FINAL}): {final_name}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    frozen = {}
    for name, res in chosen.items():
        _log(f"freezing {name}: OOF predictions for Youden threshold, then fit on all train rows")
        est = build_estimator(res.family, res.params)
        oof = oof_probabilities(est, X[name], y, cv)
        thr = youden_threshold(y, oof)
        est.fit(X[name], y)
        joblib.dump(est, MODELS_DIR / f"{name}.joblib")
        frozen[name] = {
            "family": res.family,
            "params": res.params,
            "features": list(X[name].columns),
            "log_features": [c for c in LOG_FEATURES if c in X[name].columns],
            "n_features": int(X[name].shape[1]),
            "cv_auc_mean": res.mean_auc,
            "cv_auc_std": res.std_auc,
            "cv_fold_aucs": res.fold_aucs,
            "oof_auc": float(roc_auc_score(y, oof)),
            "threshold": thr,
            "n_train": int(len(y)),
            "train_prevalence": float(y.mean()),
            "estimator_path": str((MODELS_DIR / f"{name}.joblib").relative_to(REPO)),
        }
        if name == "model_A_full":
            model_a_coefficients(est, X[name].columns).to_csv(REPORTS_DIR / "model_A_coefficients.csv", index=False)

    config = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "sklearn_version": sklearn.__version__,
        "cv": f"StratifiedKFold(n_splits={N_SPLITS}, shuffle=True, random_state={SEED}) on train cycles only",
        "selection_rule": f"highest mean CV ROC-AUC among {CANDIDATES_FOR_FINAL}",
        "final_model": final_name,
        "threshold_rule": "Youden's J on train out-of-fold predictions (never on test)",
        "models": frozen,
    }
    (REPORTS_DIR / "frozen_config.json").write_text(json.dumps(config, indent=2))

    log_lines = [
        f"## Run {config['created_utc']} - scripts/nhanes_05_train_cv.py",
        "",
        f"- train N = {len(y):,} (cycles 2005-2016), CKD prevalence = {y.mean():.4f}",
        f"- {config['cv']}; scoring = ROC-AUC; seed = {SEED}; sklearn {sklearn.__version__}",
        f"- selection rule: {config['selection_rule']}",
        "",
        "| model_set | family | params | n_features | fold AUCs | mean AUC | sd | seconds |",
        "|---|---|---|---:|---|---:|---:|---:|",
    ]
    for r in table.itertuples(index=False):
        log_lines.append(f"| {r.model_set} | {r.family} | {r.params} | {r.n_features} | {r.fold_aucs} | "
                         f"{r.mean_auc:.4f} | {r.std_auc:.4f} | {r.seconds:.1f} |")
    log_lines += ["", "Chosen per model set (frozen, fit on all train rows, threshold = Youden's J on train OOF):", ""]
    for name, spec in frozen.items():
        t = spec["threshold"]
        log_lines.append(f"- **{name}**: {spec['family']} {spec['params']} -> CV AUC {spec['cv_auc_mean']:.4f} "
                         f"(sd {spec['cv_auc_std']:.4f}); OOF AUC {spec['oof_auc']:.4f}; threshold {t['threshold']:.4f} "
                         f"(OOF sens {t['sensitivity']:.3f}, spec {t['specificity']:.3f})")
    log_lines += ["", f"**FINAL MODEL: {final_name}** (frozen before any test-set evaluation).", ""]
    with (REPORTS_DIR / "run_log.md").open("a") as fh:
        fh.write("\n".join(log_lines) + "\n")
    _log("done: cv_results.csv, frozen_config.json, model_A_coefficients.csv, run_log.md written")


if __name__ == "__main__":
    main()
