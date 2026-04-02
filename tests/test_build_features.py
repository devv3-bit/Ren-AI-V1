import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline

from src.features.build_features import build_base_features
from src.pipeline.run_pipeline import run_pipeline
from src.data.contracts import TARGET_COL


def _get_df_valid():
    result = run_pipeline("data/raw/CKD.csv")
    return result.df_valid


def test_build_features_returns_X_y():
    df_valid = _get_df_valid()
    result = build_base_features(df_valid, require_target=True)

    assert isinstance(result.X_df, pd.DataFrame)
    assert isinstance(result.preprocessor, Pipeline)
    assert isinstance(result.y, np.ndarray)
    assert result.y.dtype == int
    assert len(result.y) == result.X_df.shape[0]
    X = result.preprocessor.fit_transform(result.X_df)
    assert np.isfinite(X).all()
    assert set(np.unique(result.y)).issubset({0, 1})


def test_build_features_without_target():
    df_valid = _get_df_valid()
    df_no_target = df_valid.drop(columns=[TARGET_COL])
    result = build_base_features(df_no_target, require_target=False)

    assert result.y is None
    assert result.X_df.shape[0] == df_no_target.shape[0]
