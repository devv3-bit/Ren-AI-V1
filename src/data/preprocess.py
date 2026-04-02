import pandas as pd 
from src.config import I_SCHEMA,NUMERIC_COLS
"""
Preprocessing Layer – Column Standardization & Type Coercion

Purpose: Converts raw dataset inputs to systems's internal schema (src/data/config.py)
The goal is to produce a clean, predictable DataFrame for downstream processing.
"""


MISSING_TOKENS = {"?": None, "": None, " ": None, "\t": None}
#create a function "standardize_columns" to rename raw dataset coloumns to canonical names.
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df= df.copy()
    df.rename(columns= I_SCHEMA , inplace=True)
    return df

def coerce_types(df: pd.DataFrame)-> pd.DataFrame: 
    df=df.copy()
    df = df.replace(MISSING_TOKENS)
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

    
