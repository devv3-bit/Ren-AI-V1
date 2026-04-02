import pandas as pd

from src.data.load_data import load_raw_ckd
from src.data.preprocess import standardize_columns, coerce_types 
from src.config import I_SCHEMA, NUMERIC_COLS


def main():
    df_raw = load_raw_ckd("data/raw/ckd.csv")
    print("RAW:", df_raw.shape)
    print("RAW columns (first 10):", df_raw.columns.tolist()[:10])

    df_std = standardize_columns(df_raw)
    print("\nSTD:", df_std.shape)

    sample_raw_key = next(iter(I_SCHEMA.keys()))
    sample_canon = I_SCHEMA[sample_raw_key]
    print("Sample mapping check:", sample_raw_key, "->", sample_canon)
    print("Canon present?", sample_canon in df_std.columns)

    df_clean = coerce_types(df_std)
    print("\nCLEAN:", df_clean.shape)

    check_cols = [c for c in ["age", "blood_pressure", "serum_creatinine", "hemoglobin"] if c in df_clean.columns]
    print("\nPreview key numeric cols:")
    print(df_clean[check_cols].head())

    print("\nDtypes of key numeric cols:")
    for c in check_cols:
        print(c, "->", df_clean[c].dtype)

    non_numeric = []
    for c in NUMERIC_COLS:
        if c in df_clean.columns and not pd.api.types.is_numeric_dtype(df_clean[c]):
            non_numeric.append((c, str(df_clean[c].dtype)))

    print("\nNUMERIC_COLS dtype check:")
    if non_numeric:
        print("❌ These expected numeric columns are not numeric:", non_numeric)
    else:
        print("✅ All present NUMERIC_COLS are numeric (or NaN).")

    print("\n✅ Day 3 test complete.")
if __name__ == "__main__":
    main()
