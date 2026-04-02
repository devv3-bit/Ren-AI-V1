import pandas as pd
import pytest

def test_contract_missing_required_column_raises():
    from src.data.contracts import MODEL_REQUIRED_BASE_FEATURES, assert_required_columns

    # make a df with all required columns except one
    cols = list(MODEL_REQUIRED_BASE_FEATURES)
    missing = cols.pop()  # remove 1 required column
    df = pd.DataFrame({c: [1] for c in cols})

    with pytest.raises(ValueError):
        assert_required_columns(df, MODEL_REQUIRED_BASE_FEATURES, context="test")
