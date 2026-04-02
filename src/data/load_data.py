from pathlib import Path
import pandas as pd


def load_raw_ckd(path: str) -> pd.DataFrame:
    """
    Load raw CKD dataset from a CSV file.

    Parameters

    path : str
        Relative or absolute path to the raw CKD CSV file.

    Returns
    pd.DataFrame
        Raw CKD dataset as loaded from disk.

    Raises
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is empty or cannot be read properly.
    """

    file_path = Path(path)

    # 1) Check file exists
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    # 2) Try reading CSV
    try: 
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}")

    # 3) Basic sanity checks
    if df.empty:
        raise ValueError("Loaded dataset is empty.")

    if df.shape[1] == 0:
        raise ValueError("Dataset has no columns.")

    return df
