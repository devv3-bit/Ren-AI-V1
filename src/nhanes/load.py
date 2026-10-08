# src/nhanes/load.py
"""
Read the raw NHANES SAS transport files and build one merged table.

For each cycle the 8 component files are left-joined onto DEMO on SEQN,
a `cycle` column is added, and all cycles are stacked.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.nhanes.constants import (
    COMPONENTS,
    CYCLE_ORDER,
    MERGED_PATH,
    N_EXPECTED_FILES,
    RAW_DIR,
    xpt_basename,
    xpt_path,
)


def _decode_bytes_columns(df: pd.DataFrame) -> pd.DataFrame:
    """SAS character variables arrive as bytes; decode them so parquet/CSV can store them."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object and out[col].map(lambda v: isinstance(v, bytes)).any():
            out[col] = out[col].map(lambda v: v.decode("utf-8", "replace") if isinstance(v, bytes) else v)
    return out


def read_xpt(path: Path) -> pd.DataFrame:
    """Read one NHANES .xpt file; SEQN is returned as int64."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NHANES file not found: {path}")
    df = pd.read_sas(path, format="xport")
    if "SEQN" not in df.columns:
        raise ValueError(f"{path.name} has no SEQN column")
    df = _decode_bytes_columns(df)
    return df.assign(SEQN=df["SEQN"].astype("int64"))


def verify_raw_files(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """
    Check that all 56 files exist and are readable; return a manifest with
    row/column counts per file. Raises if any file is missing.
    """
    rows: List[dict] = []
    missing: List[str] = []
    for cycle in CYCLE_ORDER:
        for component in COMPONENTS:
            path = Path(raw_dir) / f"{xpt_basename(cycle, component)}.xpt"
            if not path.exists():
                missing.append(path.name)
                continue
            df = read_xpt(path)
            rows.append(
                {
                    "cycle": cycle,
                    "component": component,
                    "file": path.name,
                    "rows": int(len(df)),
                    "cols": int(df.shape[1]),
                    "size_mb": round(path.stat().st_size / 1e6, 2),
                    "seqn_unique": bool(df["SEQN"].is_unique),
                }
            )
    if missing:
        raise FileNotFoundError(f"{len(missing)} NHANES files missing: {missing}")
    manifest = pd.DataFrame(rows)
    if len(manifest) != N_EXPECTED_FILES:
        raise RuntimeError(f"expected {N_EXPECTED_FILES} files, found {len(manifest)}")
    return manifest


def load_cycle(cycle: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Left-join all component files of one cycle onto DEMO on SEQN."""
    merged = read_xpt(Path(raw_dir) / f"{xpt_basename(cycle, 'DEMO')}.xpt")
    for component in COMPONENTS[1:]:
        other = read_xpt(Path(raw_dir) / f"{xpt_basename(cycle, component)}.xpt")
        overlap = sorted((set(other.columns) & set(merged.columns)) - {"SEQN"})
        if overlap:
            # Keep the DEMO/earlier value; NHANES component files should not overlap.
            other = other.drop(columns=overlap)
        merged = merged.merge(other, on="SEQN", how="left", validate="one_to_one")
    return merged.assign(cycle=cycle)


def load_all_cycles(cycles: Optional[List[str]] = None, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    cycles = list(cycles) if cycles is not None else list(CYCLE_ORDER)
    frames = [load_cycle(c, raw_dir=raw_dir) for c in cycles]
    return pd.concat(frames, ignore_index=True, sort=False)


def _parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def save_table(df: pd.DataFrame, path: Path = MERGED_PATH) -> Path:
    """Save as parquet when pyarrow is available, otherwise CSV (same stem)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet" and not _parquet_available():
        path = path.with_suffix(".csv")
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    return path


def load_table(path: Path = MERGED_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() and path.suffix == ".parquet":
        path = path.with_suffix(".csv")
    if not path.exists():
        raise FileNotFoundError(f"table not found: {path} (run the preceding nhanes script)")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)
