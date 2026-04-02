# src/data/validate_labs.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd

from src.config import VALIDATION_RULES


@dataclass
class ValidationIssue:
    """Represents one flagged value in the dataset."""
    idx: Any
    col: str
    value: Any
    issue_type: str  # "soft" or "hard"
    message: str


def _get_bounds(rule: dict) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Returns (hard_min, hard_max, soft_min, soft_max) from a rule.

    Supports:
      - {"hard": (min,max), "soft": (min,max)}
      - {"hard": (min,max)}
      - {"hard_min": x, "hard_max": y}  (legacy style)
      - {"soft": (min,max)} only
    """
    hard_min = hard_max = soft_min = soft_max = None

    if "hard" in rule and rule["hard"] is not None:
        hard_min, hard_max = rule["hard"]
    else:
        # legacy
        if "hard_min" in rule:
            hard_min = rule["hard_min"]
        if "hard_max" in rule:
            hard_max = rule["hard_max"]

    if "soft" in rule and rule["soft"] is not None:
        soft_min, soft_max = rule["soft"]

    return hard_min, hard_max, soft_min, soft_max


def validate_column_bounds(df: pd.DataFrame, col: str, rule: dict) -> Tuple[List[ValidationIssue], List[ValidationIssue]]:
    """
    Validate one numeric column against its rule.

    - HARD violation -> returned as hard_issues (you typically treat as fatal)
    - SOFT violation -> returned as soft_issues (warnings)
    """
    soft_issues: List[ValidationIssue] = []
    hard_issues: List[ValidationIssue] = []

    if col not in df.columns:
        # Column not present: not an error, just skip
        return soft_issues, hard_issues

    hard_min, hard_max, soft_min, soft_max = _get_bounds(rule)

    # Only check numeric-ish values; NaN is ignored
    series = pd.to_numeric(df[col], errors="coerce")

    for idx, val in series.dropna().items():
        # HARD checks (impossible/sanity)
        if hard_min is not None and val < hard_min:
            hard_issues.append(
                ValidationIssue(idx, col, val, "hard", f"{col} below hard_min ({hard_min})")
            )
            continue
        if hard_max is not None and val > hard_max:
            hard_issues.append(
                ValidationIssue(idx, col, val, "hard", f"{col} above hard_max ({hard_max})")
            )
            continue

        # SOFT checks (clinically suspicious)
        if soft_min is not None and val < soft_min:
            soft_issues.append(
                ValidationIssue(idx, col, val, "soft", f"{col} below soft_min ({soft_min})")
            )
        elif soft_max is not None and val > soft_max:
            soft_issues.append(
                ValidationIssue(idx, col, val, "soft", f"{col} above soft_max ({soft_max})")
            )

    return soft_issues, hard_issues


def validate_labs(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate all columns listed in VALIDATION_RULES.

    """
    all_soft: List[ValidationIssue] = []
    all_hard: List[ValidationIssue] = []

    for col, rule in VALIDATION_RULES.items():
        soft, hard = validate_column_bounds(df, col, rule)
        all_soft.extend(soft)
        all_hard.extend(hard)

    summary = {
        "hard_error_count": len(all_hard),
        "soft_warning_count": len(all_soft),
        "hard_error_cols": sorted({x.col for x in all_hard}),
        "soft_warning_cols": sorted({x.col for x in all_soft}),
    }

    return {
        "hard_errors": all_hard,
        "soft_warnings": all_soft,
        "summary": summary,
    }


def raise_on_hard_errors(report: Dict[str, Any]) -> None:
    hard = report.get("hard_errors", [])
    if hard:
        preview = hard[:10]
        msg_lines = ["Hard validation errors found (showing up to 10):"]
        for e in preview:
            msg_lines.append(f"- idx={e.idx} col={e.col} value={e.value} :: {e.message}")
        raise ValueError("\n".join(msg_lines))



