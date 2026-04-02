# src/models/train_logistic.py
"""
Train a frequentist logistic regression for CKD risk.

Goals:
- Treat modeling as statistical inference
- Use frozen feature engineering
- Report performance AND coefficients
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

from src.pipeline.run_pipeline import run_pipeline
from src.features.build_features import get_base_X_df, get_y, make_preprocessor


# -----------------------------
# Configuration
# -----------------------------
DATA_PATH = "data/raw/CKD.csv"
N_SPLITS = 5
RANDOM_STATE = 42


def train_logistic_regression():
    # 1️⃣ Load validated data
    pipe_out = run_pipeline(DATA_PATH, require_target=True)
    df_valid = pipe_out.df_valid

    # 2️⃣ Build features
    X_df = get_base_X_df(df_valid)
    y = get_y(df_valid, require_target=True)
    feature_names = list(X_df.columns)

    # 3️⃣ Define model
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
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    # 4️⃣ Cross-validated predictions
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    y_prob = cross_val_predict(
        model,
        X_df,
        y,
        cv=cv,
        method="predict_proba",
    )[:, 1]

    y_pred = (y_prob >= 0.5).astype(int)

    # 5️⃣ Metrics
    auc = roc_auc_score(y, y_prob)
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred, target_names=["notCKD", "CKD"])

    print("\n=== Cross-validated Performance ===")
    print(f"ROC-AUC: {auc:.3f}")
    print("\nConfusion Matrix (threshold=0.5):")
    print(cm)
    print("\nClassification Report:")
    print(report)

    # 6️⃣ Fit once on full data for coefficients
    model.fit(X_df, y)

    imputer = model.named_steps["preprocess"].named_steps["imputer"]
    expanded_feature_names = imputer.get_feature_names_out(input_features=feature_names)

    coef = model.named_steps["clf"].coef_.ravel()
    odds_ratio = np.exp(coef)

    coef_df = pd.DataFrame(
        {
            "feature": expanded_feature_names,
            "coef_log_odds": coef,
            "odds_ratio": odds_ratio,
        }
    ).sort_values("coef_log_odds", ascending=False)

    print("\n=== Model Coefficients (Interpretation) ===")
    print(coef_df.to_string(index=False))

    return model, coef_df


if __name__ == "__main__":
    train_logistic_regression()
 
