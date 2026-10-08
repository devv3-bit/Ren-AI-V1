"""
v3 transfer model, part 1: pre-specify, tune on NHANES TRAIN only, freeze.

Context (disclosed): the v2 harmonised Model A was already evaluated once on
the 399 UCI patients (AUC 0.641) and a post-hoc diagnosis traced the gap to
standardised rare-missingness indicators and extrapolation. This script fixes
those two mechanisms, selects among candidates by NHANES CV alone, and freezes
the result BEFORE scripts/nhanes_11_transfer_v3_evaluate.py looks at UCI a
second time. The v2 number stays on record.

Writes reports/nhanes/cv_results_v3.csv, frozen_config_v3.json, run_log.md (append),
data/external/nhanes/models/transfer_*.joblib
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import joblib
import sklearn
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.nhanes.constants import (  # noqa: E402
    COHORT_PATH, HARMONIZED_FEATURES, HGB_GRID, LOGISTIC_C_GRID, LOG_FEATURES, MODELS_DIR,
    N_SPLITS, REPORTS_DIR, SEED, TARGET_COL,
)
from src.nhanes.features import build_X, split_train_test  # noqa: E402
from src.nhanes.load import load_table  # noqa: E402
from src.nhanes.models import (  # noqa: E402
    best_result, build_estimator, make_cv, oof_probabilities, results_table, tune_hgb,
    tune_logistic, youden_threshold,
)
from src.nhanes.transfer import CLIP_LOWER_QUANTILE, CLIP_UPPER_QUANTILE  # noqa: E402

CANDIDATES_FOR_FINAL = ["transfer_lr_v3", "transfer_hgb_14"]
ABLATION = "transfer_lr_v3_noclip"

PRESPEC = f"""## v3 transfer model: pre-specification (written before the second look at UCI)

- Disclosure: UCI has been evaluated once (v2 harmonised Model A, AUC 0.641, see external_metrics.json).
  The post-hoc diagnosis (external_diagnostics.json) found two mechanisms: standardised rare-missingness
  indicators (-55 log-odds for a missing potassium) and extrapolation on hospital-range values (z up to 26).
- v3 changes, fixed now: (1) no missing-value indicators; (2) features clipped to the NHANES-train
  {CLIP_LOWER_QUANTILE:.1%}-{CLIP_UPPER_QUANTILE:.1%} range before scaling; (3) a gradient-boosting candidate on
  the same 14 shared features. Same 14 features, same log policy, same label, same NHANES train rows.
- Selection rule: highest mean 5-fold CV ROC-AUC on NHANES train among {CANDIDATES_FOR_FINAL}.
  UCI plays no part in selection. `{ABLATION}` (no indicators, no clipping) is an ablation, reported only.
- Thresholds: Youden's J on NHANES-train out-of-fold predictions.
- UCI is then evaluated exactly once more by scripts/nhanes_11_transfer_v3_evaluate.py; whatever the
  number is, it is reported next to the v2 number as a disclosed second look.

"""


def _log(msg: str) -> None:
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    with (REPORTS_DIR / "run_log.md").open("a") as fh:
        fh.write(PRESPEC)

    cohort = load_table(COHORT_PATH)
    train, _ = split_train_test(cohort)
    y = train[TARGET_COL].to_numpy().astype(int)
    X = build_X(train, HARMONIZED_FEATURES, log_cols=LOG_FEATURES)
    cv = make_cv()
    _log(f"train N = {len(y):,}, {X.shape[1]} shared features; CV = StratifiedKFold({N_SPLITS}, shuffle, seed {SEED})")

    results = []
    _log("v3 logistic (no indicators + clipping), C grid")
    results += tune_logistic("transfer_lr_v3", X, y, cv, family="transfer_logistic")
    _log("ablation: no indicators, no clipping")
    results += tune_logistic(ABLATION, X, y, cv, family="transfer_logistic_noclip")
    _log(f"gradient boosting on 14 features, grid {HGB_GRID}")
    results += tune_hgb("transfer_hgb_14", X, y, cv)

    table = results_table(results)
    table.to_csv(REPORTS_DIR / "cv_results_v3.csv", index=False)
    print("\n=== v3 configurations (5-fold CV ROC-AUC on NHANES train) ===")
    print(table[["model_set", "family", "params", "mean_auc", "std_auc", "fold_aucs", "seconds"]].to_string(index=False))

    chosen = {name: best_result(results, name) for name in CANDIDATES_FOR_FINAL + [ABLATION]}
    final_name = max(CANDIDATES_FOR_FINAL, key=lambda n: chosen[n].mean_auc)
    print(f"\nFINAL v3 transfer model (NHANES CV only): {final_name}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    frozen = {}
    for name, res in chosen.items():
        _log(f"freezing {name}: OOF threshold, fit on all train rows")
        est = build_estimator(res.family, res.params)
        oof = oof_probabilities(est, X, y, cv)
        thr = youden_threshold(y, oof)
        est.fit(X, y)
        joblib.dump(est, MODELS_DIR / f"{name}.joblib")
        frozen[name] = {
            "family": res.family, "params": res.params, "features": list(X.columns),
            "log_features": [c for c in LOG_FEATURES if c in X.columns], "n_features": int(X.shape[1]),
            "cv_auc_mean": res.mean_auc, "cv_auc_std": res.std_auc, "cv_fold_aucs": res.fold_aucs,
            "oof_auc": float(roc_auc_score(y, oof)), "threshold": thr, "n_train": int(len(y)),
            "train_prevalence": float(y.mean()),
            "estimator_path": str((MODELS_DIR / f"{name}.joblib").relative_to(REPO)),
            "role": "ablation" if name == ABLATION else ("final" if name == final_name else "secondary candidate"),
        }

    config = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "version": "v3 transfer model (second look at UCI, disclosed)",
        "seed": SEED, "sklearn_version": sklearn.__version__,
        "cv": f"StratifiedKFold(n_splits={N_SPLITS}, shuffle=True, random_state={SEED}) on NHANES train only",
        "selection_rule": f"highest mean CV ROC-AUC among {CANDIDATES_FOR_FINAL}; UCI not used",
        "final_model": final_name,
        "clip_quantiles": [CLIP_LOWER_QUANTILE, CLIP_UPPER_QUANTILE],
        "models": frozen,
    }
    (REPORTS_DIR / "frozen_config_v3.json").write_text(json.dumps(config, indent=2))

    lines = [f"## v3 training run {config['created_utc']} - scripts/nhanes_10_transfer_v3_train.py", "",
             "| model_set | family | params | fold AUCs | mean AUC | sd |", "|---|---|---|---|---:|---:|"]
    for r in table.itertuples(index=False):
        lines.append(f"| {r.model_set} | {r.family} | {r.params} | {r.fold_aucs} | {r.mean_auc:.4f} | {r.std_auc:.4f} |")
    lines += [""]
    for name, spec in frozen.items():
        t = spec["threshold"]
        lines.append(f"- **{name}** ({spec['role']}): {spec['family']} {spec['params']} -> CV AUC {spec['cv_auc_mean']:.4f}; "
                     f"threshold {t['threshold']:.4f} (OOF sens {t['sensitivity']:.3f}, spec {t['specificity']:.3f})")
    lines += ["", f"**FINAL v3 transfer model: {final_name}** (frozen before the second UCI evaluation).", ""]
    with (REPORTS_DIR / "run_log.md").open("a") as fh:
        fh.write("\n".join(lines) + "\n")
    _log("done: cv_results_v3.csv, frozen_config_v3.json, run_log.md")


if __name__ == "__main__":
    main()
