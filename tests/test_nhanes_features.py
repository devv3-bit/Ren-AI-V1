import numpy as np
import pandas as pd
import pytest

from src.nhanes.constants import (
    BLOOD_TEST_FEATURES,
    ENGINEERED_FEATURES,
    HARMONIZED_FEATURES,
    MODEL_FEATURES,
    URINE_COLS,
)
from src.nhanes.features import (
    add_engineered_features,
    build_X,
    feature_inventory,
    split_train_test,
)


def test_urine_variables_are_never_features():
    assert not set(URINE_COLS) & set(MODEL_FEATURES)
    assert not set(URINE_COLS) & set(HARMONIZED_FEATURES)
    with pytest.raises(ValueError):
        build_X(pd.DataFrame({"acr": [1.0], "age": [40.0]}), feature_cols=["age", "acr"])


def test_feature_counts_for_the_application():
    inv = feature_inventory()
    assert inv["n_blood_tests"] == 17 == len(set(BLOOD_TEST_FEATURES))
    assert inv["n_engineered_scores"] == 2 == len(ENGINEERED_FEATURES)
    assert inv["n_model_features"] == 25
    assert set(HARMONIZED_FEATURES) <= set(MODEL_FEATURES)


def test_engineered_ratio_and_log_transform():
    df = pd.DataFrame({"bun": [14.0, 20.0, np.nan], "serum_creatinine": [1.0, 0.0, 1.0]})
    out = add_engineered_features(df)
    assert out.loc[0, "bun_creatinine_ratio"] == 14.0
    assert np.isnan(out.loc[1, "bun_creatinine_ratio"])  # divide by zero -> NaN
    assert "bun_creatinine_ratio" not in df.columns  # input not mutated

    X = build_X(out, feature_cols=["bun", "serum_creatinine", "bun_creatinine_ratio"])
    assert X.loc[0, "bun"] == pytest.approx(np.log1p(14.0))
    assert X.loc[0, "serum_creatinine"] == pytest.approx(np.log1p(1.0))
    assert np.isnan(X.loc[2, "bun"])
    assert list(X.columns) == ["bun", "serum_creatinine", "bun_creatinine_ratio"]


def test_split_train_test_by_cycle():
    df = pd.DataFrame({"cycle": ["D", "E", "F", "G", "H", "I", "P", "P"], "x": range(8)})
    train, test = split_train_test(df)
    assert len(train) == 6 and len(test) == 2
    assert set(test["cycle"]) == {"P"}
    with pytest.raises(ValueError):
        split_train_test(pd.DataFrame({"cycle": ["J"], "x": [0]}))
