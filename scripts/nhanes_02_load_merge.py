"""
Step 2: read the 8 component files of each NHANES cycle, left-join them onto
DEMO on SEQN, add a `cycle` column, stack all cycles and save the merged raw
table to data/external/nhanes/merged.parquet (CSV fallback).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.nhanes.constants import CYCLES, CYCLE_ORDER, MERGED_PATH  # noqa: E402
from src.nhanes.load import load_all_cycles, save_table  # noqa: E402


def main() -> None:
    merged = load_all_cycles()
    counts = merged["cycle"].value_counts()
    print("rows per cycle (= DEMO rows, left join):")
    for cycle in CYCLE_ORDER:
        print(f"  {cycle}  {CYCLES[cycle]['label']:<28s} {int(counts.get(cycle, 0)):>7,d}")
    print(f"  total {int(len(merged)):>35,d}")
    print(f"columns in merged table: {merged.shape[1]}")
    path = save_table(merged, MERGED_PATH)
    print(f"saved -> {path.relative_to(REPO)}  shape={merged.shape}")


if __name__ == "__main__":
    main()
