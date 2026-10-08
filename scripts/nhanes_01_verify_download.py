"""
Step 1 check: confirm all 56 NHANES .xpt files are present and readable with
pd.read_sas(path, format="xport"); print row counts per file and write a
manifest to reports/nhanes/raw_file_manifest.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.nhanes.constants import N_EXPECTED_FILES, REPORTS_DIR  # noqa: E402
from src.nhanes.load import verify_raw_files  # noqa: E402


def main() -> None:
    manifest = verify_raw_files()
    print(manifest.to_string(index=False))
    print(f"\n{len(manifest)}/{N_EXPECTED_FILES} files readable; "
          f"all SEQN unique: {bool(manifest['seqn_unique'].all())}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "raw_file_manifest.md"
    lines = [
        "# NHANES raw file manifest",
        "",
        "Produced by `scripts/nhanes_01_verify_download.py`; files fetched by "
        "`scripts/download_nhanes.sh` from `https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{YEAR}/DataFiles/`.",
        "",
        "| cycle | component | file | rows | cols | size (MB) | SEQN unique |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for r in manifest.itertuples(index=False):
        lines.append(f"| {r.cycle} | {r.component} | {r.file} | {r.rows} | {r.cols} | {r.size_mb} | {r.seqn_unique} |")
    lines += ["", f"Total: {len(manifest)} files, {int(manifest['rows'].sum())} rows across all files."]
    out.write_text("\n".join(lines) + "\n")
    print(f"manifest written to {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
