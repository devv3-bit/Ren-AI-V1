"""
Steps 4-7: cohort flow, KDIGO label, engineered features, temporal split.

Writes
  data/external/nhanes/cohort.parquet   analytic cohort with labels + features
  reports/nhanes/cohort_flow.md         exact counts at every filter step
  reports/nhanes/cohort_summary.json    numbers reused by the write-up
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.nhanes.cohort import apply_cohort_filters, flow_markdown, flow_table, split_label  # noqa: E402
from src.nhanes.constants import COHORT_PATH, HARMONIZED_PATH, REPORTS_DIR  # noqa: E402
from src.nhanes.features import add_engineered_features, feature_inventory  # noqa: E402
from src.nhanes.harmonize import plausibility_report  # noqa: E402
from src.nhanes.label import add_egfr, add_labels, prevalence_by_group, prevalence_summary  # noqa: E402
from src.nhanes.load import load_table, save_table  # noqa: E402


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def main() -> None:
    h = load_table(HARMONIZED_PATH)
    cohort, steps = apply_cohort_filters(h)
    cohort = add_engineered_features(add_labels(add_egfr(cohort)))
    cohort = cohort.assign(split=split_label(cohort["cycle"]))

    print("=== cohort flow ===")
    print(flow_table(steps).to_string(index=False))

    overall = prevalence_summary(cohort)
    by_split = prevalence_by_group(cohort, "split").set_index("split").reindex(["train", "test"])
    by_cycle = prevalence_by_group(cohort, "cycle")

    print("\n=== label (KDIGO single-visit: eGFR < 60 OR ACR >= 30) ===")
    print(f"N = {overall['n']:,}   CKD = {overall['n_ckd']:,}   prevalence = {_pct(overall['prevalence'])}")
    print(f"  eGFR < 60 only: {overall['n_egfr_low'] - overall['n_both']:,}   "
          f"ACR >= 30 only: {overall['n_albuminuria'] - overall['n_both']:,}   both: {overall['n_both']:,}")
    print(f"  early subgroup (eGFR >= 60): n = {overall['n_early_subgroup']:,}, "
          f"CKD (albuminuria only) = {overall['n_ckd_early_subgroup']:,}, "
          f"prevalence = {_pct(overall['prevalence_early_subgroup'])}")
    print("\n=== by split (temporal) ===")
    print(by_split[["n", "n_ckd", "prevalence", "n_early_subgroup", "prevalence_early_subgroup"]].to_string())
    print("\n=== by cycle ===")
    print(by_cycle[["cycle", "n", "n_ckd", "prevalence", "n_early_subgroup", "prevalence_early_subgroup"]].to_string(index=False))

    plaus = plausibility_report(cohort)
    print("\n=== plausibility on the adult cohort (hard-range violations) ===")
    print(plaus[plaus["present"]][["column", "n_checked", "hard_range", "hard_violations"]].to_string(index=False))

    inv = feature_inventory()
    print(f"\nfeatures: {inv['n_blood_tests']} blood tests + {inv['n_vitals_demographics']} vitals/demographics "
          f"+ {inv['n_engineered_scores']} engineered scores = {inv['n_model_features']} (+ missing indicators)")

    out_path = save_table(cohort, COHORT_PATH)
    print(f"\nsaved -> {out_path.relative_to(REPO)} shape={cohort.shape}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    md = flow_markdown(steps, cohort)
    md += "\n".join([
        "",
        "## Label (Step 5)",
        "",
        "KDIGO-style, single visit: `ckd = 1` if eGFR (CKD-EPI 2021, race-free, recorded sex) < 60 mL/min/1.73 m2 "
        "OR urine ACR >= 30 mg/g. NHANES has no 3-month confirmation, so chronicity cannot be verified.",
        "",
        "| group | n | CKD | prevalence | early subgroup (eGFR >= 60) n | CKD in early subgroup | prevalence |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| all | {overall['n']:,} | {overall['n_ckd']:,} | {_pct(overall['prevalence'])} | "
        f"{overall['n_early_subgroup']:,} | {overall['n_ckd_early_subgroup']:,} | {_pct(overall['prevalence_early_subgroup'])} |",
    ] + [
        f"| {idx} | {int(r.n):,} | {int(r.n_ckd):,} | {_pct(r.prevalence)} | {int(r.n_early_subgroup):,} | "
        f"{int(r.n_ckd_early_subgroup):,} | {_pct(r.prevalence_early_subgroup)} |"
        for idx, r in by_split.iterrows()
    ] + [
        "",
        f"Label composition (all): eGFR < 60 only {overall['n_egfr_low'] - overall['n_both']:,}; "
        f"ACR >= 30 only {overall['n_albuminuria'] - overall['n_both']:,}; both {overall['n_both']:,}.",
        "",
        "## Plausibility on the adult cohort",
        "",
        "| column | n checked | hard range | violations |", "|---|---:|---|---:|",
    ] + [
        f"| {r.column} | {r.n_checked:,} | {r.hard_range} | {r.hard_violations} |"
        for r in plaus[plaus["present"]].itertuples(index=False)
    ] + [""])
    (REPORTS_DIR / "cohort_flow.md").write_text(md)

    summary = {
        "flow": [s.__dict__ for s in steps],
        "final_n": steps[-1].n_remaining,
        "overall": overall,
        "by_split": {k: {kk: (float(vv) if isinstance(vv, float) else int(vv)) for kk, vv in v.items()}
                     for k, v in by_split.to_dict(orient="index").items()},
        "by_cycle": by_cycle.to_dict(orient="records"),
        "feature_inventory": inv,
        "plausibility_cohort": plaus.assign(hard_range=plaus["hard_range"].astype(str)).to_dict(orient="records"),
    }
    (REPORTS_DIR / "cohort_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"wrote {(REPORTS_DIR / 'cohort_flow.md').relative_to(REPO)} and cohort_summary.json")


if __name__ == "__main__":
    main()
