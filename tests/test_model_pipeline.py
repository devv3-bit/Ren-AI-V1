import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from src.features.build_features import get_base_X_df, get_y, make_preprocessor
from src.pipeline.run_pipeline import run_pipeline


def test_logistic_pipeline_cross_val_predictions():
    df_valid = run_pipeline("data/raw/CKD.csv").df_valid
    X_df = get_base_X_df(df_valid)
    y = get_y(df_valid, require_target=True)

    assert set(np.unique(y)).issubset({0, 1})

    model = Pipeline(
        steps=[
            ("preprocess", make_preprocessor(add_missing_indicators=True)),
            (
                "clf",
                LogisticRegression(
                    penalty="l2",
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    y_prob = cross_val_predict(model, X_df, y, cv=cv, method="predict_proba")[:, 1]

    assert np.isfinite(y_prob).all()
    auc = roc_auc_score(y, y_prob)
    assert np.isfinite(auc)
