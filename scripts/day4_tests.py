# scripts/day4_tests.py

from __future__ import annotations

from collections import Counter
from typing import Any


def _get(obj: Any, name: str):
    return getattr(obj, name, None)


def _as_issue_dicts(issues: Any) -> list[dict]:
    if issues is None:
        return []

    if isinstance(issues, dict):
        out: list[dict] = []
        for k, v in issues.items():
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        out.append(item)
                    else:
                        out.append({"severity": str(k), "item": item})
            elif isinstance(v, dict):
                v2 = dict(v)
                v2.setdefault("severity", str(k))
                out.append(v2)
            else:
                out.append({"severity": str(k), "item": v})
        return out

    if isinstance(issues, (list, tuple)):
        if not issues:
            return []
        out: list[dict] = []
        for item in issues:
            if isinstance(item, dict):
                out.append(item)
            elif isinstance(item, (list, tuple)):
                if len(item) == 0:
                    out.append({"severity": "unknown", "item": item})
                elif len(item) == 2:
                    out.append({"severity": "warning", "index": item[0], "value": item[1]})
                else:
                    out.append({"severity": "warning", "item": item})
            else:
                out.append({"severity": "warning", "item": item})
        return out

    return [{"severity": "unknown", "item": issues}]


def main() -> None:
    from src.data.load_data import load_raw_ckd
    import src.data.preprocess as pp
    import src.data.validate_labs as vl

    df_raw = load_raw_ckd("data/raw/ckd.csv")

    std_fn = _get(pp, "standardize_columns")
    coerce_fn = _get(pp, "coerce_type") or _get(pp, "coerce_types")
    if std_fn is None or coerce_fn is None:
        raise ImportError("preprocess.py must expose standardize_columns and coerce_type/coerce_types")

    df = coerce_fn(std_fn(df_raw))

    validate_fn = _get(vl, "validate_labs") or _get(vl, "validate_all")
    if validate_fn is None:
        raise ImportError("validate_labs.py must expose validate_labs(df) (or validate_all(df))")

    issues_raw = validate_fn(df)
    issues = _as_issue_dicts(issues_raw)

    severities = [str(x.get("severity", "unknown")) for x in issues]
    counts = Counter(severities)

    print(f"rows={len(df)} cols={len(df.columns)}")
    print(f"issues={len(issues)}")
    print(dict(counts))

    for x in issues[:10]:
        sev = x.get("severity", "unknown")
        field = x.get("field", x.get("col", x.get("column", "")))
        msg = x.get("message", x.get("reason", ""))
        val = x.get("value", x.get("val", x.get("item", "")))
        idx = x.get("index", x.get("row", ""))
        parts = [f"[{sev}]"]
        if field:
            parts.append(str(field))
        if idx != "":
            parts.append(f"row={idx}")
        if val != "":
            parts.append(f"val={val}")
        if msg:
            parts.append(f"msg={msg}")
        print(" | ".join(parts))


if __name__ == "__main__":
    main()
