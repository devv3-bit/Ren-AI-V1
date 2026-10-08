"""
Step 3: harmonise the merged NHANES table into the repo's canonical schema,
run the biological plausibility rules (src/config.py VALIDATION_RULES via
src/data/validate_labs.py) and write reports/nhanes/harmonization_report.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.nhanes.constants import (  # noqa: E402
    CREATININE_RECAL, CYCLE_ORDER, HARMONIZED_PATH, MERGED_PATH, REPORTS_DIR,
)
from src.nhanes.harmonize import harmonize, plausibility_report  # noqa: E402
from src.nhanes.load import load_table, save_table  # noqa: E402

KEY_COLS = [
    "age", "sex", "pregnant", "serum_creatinine", "acr", "bun", "blood_glucose_random",
    "sodium", "potassium", "chloride", "bicarbonate", "serum_albumin", "uric_acid",
    "calcium", "phosphorus", "hemoglobin", "packed_cell_volume", "red_blood_cell_count",
    "white_blood_cell_count", "platelets", "hba1c", "bmi", "diabetes_mellitus",
    "sbp", "blood_pressure", "mec_weight",
]


def _md_table(df: pd.DataFrame, index_label: str = "") -> str:
    cols = [index_label] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        cells = [str(idx)] + [("" if pd.isna(v) else (f"{v:.4g}" if isinstance(v, float) else str(v))) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    merged = load_table(MERGED_PATH)
    h = harmonize(merged)
    out_path = save_table(h, HARMONIZED_PATH)

    availability = h.groupby("cycle")[KEY_COLS].count().reindex(CYCLE_ORDER)
    acr_source = pd.crosstab(h["cycle"], h["acr_source"].fillna("missing")).reindex(CYCLE_ORDER)
    d_rows = h[h["cycle"].eq(str(CREATININE_RECAL["cycle"]))]
    recal = d_rows[["serum_creatinine_raw", "serum_creatinine"]].describe().T
    plaus = plausibility_report(h)

    print(f"harmonized rows: {len(h):,}  columns: {h.shape[1]}  -> {out_path.relative_to(REPO)}")
    print("\nnon-missing counts per cycle:")
    print(availability.T.to_string())
    print("\nACR source per cycle:")
    print(acr_source.to_string())
    print(f"\n2005-06 creatinine recalibration ({CREATININE_RECAL['intercept']} + {CREATININE_RECAL['slope']} x raw):")
    print(recal.to_string())
    print("\nbiological plausibility (hard ranges from src/config.py VALIDATION_RULES):")
    print(plaus.to_string(index=False))
    print(f"\nTOTAL hard-range violations: {int(plaus['hard_violations'].sum())}  "
          f"soft warnings: {int(plaus['soft_warnings'].sum())}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    md = [
        "# NHANES harmonisation report",
        "",
        "Produced by `scripts/nhanes_03_harmonize.py` from `data/external/nhanes/merged.parquet`.",
        "",
        f"Rows: **{len(h):,}** participants (every DEMO row of the 7 cycles); columns: {h.shape[1]}.",
        "",
        "## Rules applied",
        "",
        "- Direct renames per `src/nhanes/constants.py::DIRECT_MAP` (canonical names match `src/config.py`).",
        "- `sex`: RIAGENDR 1 -> male, 2 -> female; `sex_male` numeric 1/0.",
        "- `pregnant`: RIDEXPRG 1 -> 1, 2 -> 0, 3 (cannot ascertain) -> missing.",
        "- `diabetes_mellitus`: DIQ010 1 -> 1, 2 -> 0, 3 (borderline) -> 0, 7/9 (refused/don't know) -> missing.",
        f"- `serum_creatinine`: 2005-06 (cycle D) only, CDC Deming recalibration "
        f"`{CREATININE_RECAL['intercept']} + {CREATININE_RECAL['slope']} x LBXSCR` "
        "(confirmed in the BIOPRO_D analytic notes: \"Standard creatinine (mg/dL) = -0.016 + 0.978 X "
        "(NHANES 05-06 uncalibrated serum creatinine, mg/dL)\"). Raw value kept as `serum_creatinine_raw`.",
        "- `white_blood_cell_count`: LBXWBCSI x 1000 (NHANES 1000 cells/uL -> cells/uL, the UCI scale).",
        "- `sbp` / `blood_pressure` (diastolic): mean of the available readings BPXSY1-4 / BPXDI1-4 "
        "(BPXOSY1-3 / BPXODI1-3 for the 2017-2020 oscillometric protocol); readings of 0 are treated as missing "
        "(none occur in these files, so this is a no-op safeguard).",
        "- `acr` (mg/g): URDACT where reported (2009-10 onward); 2005-06 and 2007-08 have no URDACT, so "
        "ACR = URXUMA (ug/mL) / URXUCR (mg/dL) x 100. Only the first urine collection is used.",
        "- `mec_weight`: WTMEC2YR (2-year cycles) or WTMECPRP (2017-2020).",
        "",
        "## Non-missing counts per cycle",
        "",
        _md_table(availability.T, "column"),
        "",
        "## ACR source per cycle",
        "",
        _md_table(acr_source, "cycle"),
        "",
        "## 2005-06 serum creatinine before/after recalibration",
        "",
        _md_table(recal, "variable"),
        "",
        "## Biological plausibility (hard ranges from `src/config.py::VALIDATION_RULES`)",
        "",
        "Values are reported, not removed, to keep the cohort definition exact. "
        "`blood_urea` is not present because NHANES reports BUN (`bun`), not urea.",
        "",
        _md_table(plaus.set_index("column"), "column"),
        "",
        f"**Total hard-range violations: {int(plaus['hard_violations'].sum())}** "
        f"(soft warnings: {int(plaus['soft_warnings'].sum())}).",
        "",
    ]
    (REPORTS_DIR / "harmonization_report.md").write_text("\n".join(md))
    print(f"report -> {(REPORTS_DIR / 'harmonization_report.md').relative_to(REPO)}")


if __name__ == "__main__":
    main()
