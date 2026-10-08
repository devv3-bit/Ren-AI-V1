# src/nhanes/cohort.py
"""
Cohort definition (Step 4).

Filters are applied in a fixed order and the count after each step is
recorded, so the final N is exact and auditable:

  0. All participants in the 7 NHANES cycles (every DEMO row)
  1. Age >= 18 years
  2. Not pregnant at exam (RIDEXPRG = 1 excluded; unknown / not asked kept)
  3. Non-missing serum creatinine AND non-missing urine ACR
     (both are required to define the KDIGO label)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd

from src.nhanes.constants import ADULT_AGE, CYCLES, CYCLE_ORDER, TEST_CYCLES, TRAIN_CYCLES

REQUIRED_COLS = ("cycle", "age", "pregnant", "serum_creatinine", "acr")


@dataclass(frozen=True)
class FlowStep:
    step: int
    description: str
    n_remaining: int
    n_excluded: int


def _step(step: int, description: str, before: pd.DataFrame, after: pd.DataFrame) -> FlowStep:
    return FlowStep(step, description, int(len(after)), int(len(before) - len(after)))


def apply_cohort_filters(h: pd.DataFrame) -> Tuple[pd.DataFrame, List[FlowStep]]:
    """Return (cohort, flow steps). The input frame is not modified."""
    missing = [c for c in REQUIRED_COLS if c not in h.columns]
    if missing:
        raise ValueError(f"harmonized table is missing columns: {missing}")

    steps: List[FlowStep] = [
        FlowStep(0, "All participants in the 7 NHANES cycles (DEMO rows)", int(len(h)), 0)
    ]
    adults = h[pd.to_numeric(h["age"], errors="coerce") >= ADULT_AGE]
    steps.append(_step(1, f"Age >= {ADULT_AGE} years", h, adults))

    not_pregnant = adults[~adults["pregnant"].eq(1.0)]
    steps.append(_step(2, "Not pregnant at exam (RIDEXPRG = 1 excluded)", adults, not_pregnant))

    labelled = not_pregnant[not_pregnant["serum_creatinine"].notna() & not_pregnant["acr"].notna()]
    steps.append(_step(3, "Non-missing serum creatinine AND non-missing urine ACR", not_pregnant, labelled))

    return labelled.reset_index(drop=True), steps


def split_label(cycle: pd.Series) -> pd.Series:
    """'train' for 2005-2016 cycles, 'test' for 2017-2020."""
    return cycle.astype(str).map(
        {**{c: "train" for c in TRAIN_CYCLES}, **{c: "test" for c in TEST_CYCLES}}
    )


def flow_table(steps: List[FlowStep]) -> pd.DataFrame:
    return pd.DataFrame([s.__dict__ for s in steps])


def cycle_table(cohort: pd.DataFrame) -> pd.DataFrame:
    counts = cohort["cycle"].astype(str).value_counts()
    rows = [
        {
            "cycle": c,
            "years": CYCLES[c]["label"],
            "split": "train" if c in TRAIN_CYCLES else "test",
            "n": int(counts.get(c, 0)),
        }
        for c in CYCLE_ORDER
    ]
    return pd.DataFrame(rows)


def flow_markdown(steps: List[FlowStep], cohort: pd.DataFrame) -> str:
    lines = [
        "# NHANES cohort flow",
        "",
        "Produced by `scripts/nhanes_04_cohort_label.py`. Filters are applied in this order.",
        "",
        "| step | criterion | excluded | remaining |",
        "|---:|---|---:|---:|",
    ]
    for s in steps:
        lines.append(f"| {s.step} | {s.description} | {s.n_excluded:,} | **{s.n_remaining:,}** |")
    final_n = steps[-1].n_remaining
    lines += ["", f"**Final analytic cohort: N = {final_n:,} adults.**", "", "## By cycle", "",
              "| cycle | years | split | n |", "|---|---|---|---:|"]
    for r in cycle_table(cohort).itertuples(index=False):
        lines.append(f"| {r.cycle} | {r.years} | {r.split} | {r.n:,} |")
    split = split_label(cohort["cycle"]).value_counts()
    lines += ["", f"Train (2005-2016): **{int(split.get('train', 0)):,}**; "
              f"Test (2017-Mar 2020): **{int(split.get('test', 0)):,}**.", ""]
    return "\n".join(lines)
