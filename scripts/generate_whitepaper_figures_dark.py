"""
Generate dark-theme whitepaper figures for RenAI v1.
Black background, neon accent palette: blue, purple, yellow, orange.
Output: whitepaper/figures/
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.pipeline.run_pipeline import run_pipeline
from src.features.build_features import (
    get_base_X_df, get_y, make_preprocessor, _safe_log1p,
    DEFAULT_LOG_COLS,
)
from src.features.derive import egfr_cr

OUT_DIR = REPO / "whitepaper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Dark neon palette ─────────────────────────────────────────────────
NEON_BLUE = "#00D4FF"
NEON_PURPLE = "#B44CFF"
NEON_YELLOW = "#FFE600"
NEON_ORANGE = "#FF8C00"

BG_BLACK = "#000000"
AXES_BG = "#0A0A0A"
GRID_COLOR = "#1A1A2E"
TEXT_WHITE = "#FFFFFF"
TEXT_DIM = "#888899"
EDGE_COLOR = "#222233"

RISK_COLOR = NEON_ORANGE
PROTECT_COLOR = NEON_BLUE

plt.rcParams.update({
    "figure.facecolor": BG_BLACK,
    "axes.facecolor": AXES_BG,
    "axes.edgecolor": EDGE_COLOR,
    "axes.grid": True,
    "grid.color": GRID_COLOR,
    "grid.alpha": 0.5,
    "grid.linewidth": 0.4,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.labelcolor": TEXT_WHITE,
    "axes.titlecolor": TEXT_WHITE,
    "xtick.color": TEXT_WHITE,
    "ytick.color": TEXT_WHITE,
    "text.color": TEXT_WHITE,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.facecolor": BG_BLACK,
    "savefig.bbox": "tight",
    "legend.facecolor": "#111122",
    "legend.edgecolor": EDGE_COLOR,
    "legend.labelcolor": TEXT_WHITE,
})


def get_model_outputs():
    pipe_out = run_pipeline("data/raw/CKD.csv", require_target=True)
    df_valid = pipe_out.df_valid

    X_df = get_base_X_df(df_valid)
    y = get_y(df_valid, require_target=True)

    model = Pipeline([
        ("preprocess", make_preprocessor(add_missing_indicators=True)),
        ("clf", LogisticRegression(
            C=1.0, l1_ratio=0, solver="liblinear",
            class_weight="balanced", max_iter=1000, random_state=42,
        )),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_prob = cross_val_predict(model, X_df, y, cv=cv, method="predict_proba")[:, 1]

    model.fit(X_df, y)
    feature_names = list(X_df.columns)
    imputer = model.named_steps["preprocess"].named_steps["imputer"]
    expanded_names = imputer.get_feature_names_out(input_features=feature_names)
    coef = model.named_steps["clf"].coef_.ravel()

    coef_df = pd.DataFrame({
        "feature": expanded_names,
        "coef": coef,
        "odds_ratio": np.exp(coef),
    }).sort_values("coef", key=abs, ascending=False)

    return df_valid, X_df, y, y_prob, coef_df


def fig6_missingness_bar(df_valid, y):
    features = [
        "red_blood_cell_count", "white_blood_cell_count", "potassium",
        "sodium", "packed_cell_volume", "hemoglobin", "sugar",
        "specific_gravity", "albumin", "blood_glucose_random",
        "blood_urea", "serum_creatinine", "blood_pressure", "age",
    ]
    features = [f for f in features if f in df_valid.columns]

    ckd_mask = y == 1
    miss_ckd = [df_valid.loc[ckd_mask, f].isna().mean() * 100 for f in features]
    miss_notckd = [df_valid.loc[~ckd_mask, f].isna().mean() * 100 for f in features]

    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = np.arange(len(features))
    h = 0.35
    ax.barh(y_pos + h/2, miss_ckd, h, color=NEON_BLUE, label="CKD", alpha=0.85,
            edgecolor=BG_BLACK)
    ax.barh(y_pos - h/2, miss_notckd, h, color=NEON_ORANGE, label="non-CKD", alpha=0.85,
            edgecolor=BG_BLACK)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f.replace("_", " ").title() for f in features], fontsize=9)
    ax.set_xlabel("Missing (%)")
    ax.set_title("Figure 6. Missingness Rate by Feature and CKD Status", color=TEXT_WHITE)
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig06_missingness.png", facecolor=BG_BLACK)
    plt.close()
    print("  -> fig06_missingness.png")


def fig7_sigmoid():
    z = np.linspace(-6, 6, 300)
    p = 1 / (1 + np.exp(-z))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(z, p, color=NEON_PURPLE, linewidth=2.5)
    ax.axhline(0.5, color=TEXT_DIM, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(0, color=TEXT_DIM, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.fill_between(z, p, 0.5, where=(p > 0.5), alpha=0.15, color=NEON_ORANGE)
    ax.fill_between(z, p, 0.5, where=(p < 0.5), alpha=0.15, color=NEON_BLUE)
    ax.set_xlabel(r"Linear predictor  $\mathbf{x}^T\boldsymbol{\beta}$")
    ax.set_ylabel(r"$P(\mathrm{CKD} \mid \mathbf{x})$")
    ax.set_title(r"Figure 7. The Sigmoid Function: $\sigma(z) = \frac{1}{1 + e^{-z}}$",
                 color=TEXT_WHITE)
    ax.set_ylim(-0.02, 1.02)
    ax.annotate("Decision boundary\n(P = 0.5)", xy=(0, 0.5), xytext=(2, 0.3),
                fontsize=9, color=TEXT_WHITE,
                arrowprops=dict(arrowstyle="->", color=TEXT_WHITE, linewidth=1.2))
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig07_sigmoid.png", facecolor=BG_BLACK)
    plt.close()
    print("  -> fig07_sigmoid.png")


def fig8_egfr_curves():
    scr = np.linspace(0.3, 10, 200)
    ages = [30, 50, 70]
    sexes = ["female", "male"]

    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {"female": "--", "male": "-"}
    colors = {30: NEON_BLUE, 50: NEON_PURPLE, 70: NEON_ORANGE}

    for age in ages:
        for sex in sexes:
            egfr_vals = [egfr_cr(s, age, sex) for s in scr]
            label = f"{sex.capitalize()}, age {age}"
            ax.plot(scr, egfr_vals, linestyle=styles[sex], color=colors[age],
                    linewidth=1.8, label=label)

    ax.axhline(60, color=NEON_YELLOW, linestyle=":", linewidth=1.2, alpha=0.8)
    ax.annotate("eGFR = 60 (CKD threshold)", xy=(7, 63), fontsize=8, color=TEXT_WHITE)
    ax.set_xlabel("Serum Creatinine (mg/dL)")
    ax.set_ylabel("eGFR (mL/min/1.73 m$^2$)")
    ax.set_title("Figure 8. CKD-EPI 2021 eGFR vs. Serum Creatinine", color=TEXT_WHITE)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(0.3, 10)
    ax.set_ylim(0, 160)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig08_egfr_curves.png", facecolor=BG_BLACK)
    plt.close()
    print("  -> fig08_egfr_curves.png")


def fig9_log_transform(df_valid):
    raw = pd.to_numeric(df_valid["serum_creatinine"], errors="coerce").dropna()
    transformed = np.log1p(np.maximum(raw.values, 0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.hist(raw, bins=40, color=NEON_BLUE, alpha=0.75, edgecolor=BG_BLACK)
    ax1.set_title("A) Raw Distribution", color=TEXT_WHITE)
    ax1.set_xlabel("Serum Creatinine (mg/dL)")
    ax1.set_ylabel("Count")

    ax2.hist(transformed, bins=40, color=NEON_ORANGE, alpha=0.75, edgecolor=BG_BLACK)
    ax2.set_title("B) After log(1 + x) Transform", color=TEXT_WHITE)
    ax2.set_xlabel("log(1 + Serum Creatinine)")
    ax2.set_ylabel("Count")

    fig.suptitle("Figure 9. Log Transform Reduces Right Skew", y=1.02,
                 fontsize=13, color=TEXT_WHITE)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig09_log_transform.png", facecolor=BG_BLACK, bbox_inches="tight")
    plt.close()
    print("  -> fig09_log_transform.png")


def fig12_roc(y, y_prob):
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color=NEON_BLUE, linewidth=2.5,
            label=f"RenAI v1 (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color=TEXT_DIM, linestyle="--", linewidth=1,
            label="Random (AUC = 0.500)")
    ax.fill_between(fpr, tpr, alpha=0.12, color=NEON_BLUE)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("Figure 12. ROC Curve (5-Fold Stratified CV)", color=TEXT_WHITE)
    ax.legend(loc="lower right")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_aspect("equal")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig12_roc.png", facecolor=BG_BLACK)
    plt.close()
    print("  -> fig12_roc.png")


def fig13_precision_recall(y, y_prob):
    precision, recall, _ = precision_recall_curve(y, y_prob)
    ap = average_precision_score(y, y_prob)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(recall, precision, color=NEON_PURPLE, linewidth=2.5,
            label=f"RenAI v1 (AP = {ap:.3f})")
    baseline = y.sum() / len(y)
    ax.axhline(baseline, color=TEXT_DIM, linestyle="--", linewidth=1,
               label=f"Baseline (prevalence = {baseline:.2f})")
    ax.fill_between(recall, precision, alpha=0.12, color=NEON_PURPLE)
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title("Figure 13. Precision-Recall Curve", color=TEXT_WHITE)
    ax.legend(loc="lower left")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig13_precision_recall.png", facecolor=BG_BLACK)
    plt.close()
    print("  -> fig13_precision_recall.png")


def fig14_calibration(y, y_prob):
    prob_true, prob_pred = calibration_curve(y, y_prob, n_bins=10, strategy="uniform")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(prob_pred, prob_true, "o-", color=NEON_YELLOW, linewidth=2,
            markersize=8, markeredgecolor=BG_BLACK, label="RenAI v1")
    ax.plot([0, 1], [0, 1], "--", color=TEXT_DIM, linewidth=1,
            label="Perfectly calibrated")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Fraction of CKD")
    ax.set_title("Figure 14. Calibration Plot", color=TEXT_WHITE)
    ax.legend(loc="lower right")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_aspect("equal")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig14_calibration.png", facecolor=BG_BLACK)
    plt.close()
    print("  -> fig14_calibration.png")


def fig15_coefficient_bar(coef_df):
    df = coef_df.copy()
    df["feature_clean"] = df["feature"].str.replace("missingindicator_", "MISSING: ")
    df["feature_clean"] = df["feature_clean"].str.replace("_", " ").str.title()
    df = df.sort_values("coef")

    colors = [NEON_ORANGE if c > 0 else NEON_BLUE for c in df["coef"]]

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(range(len(df)), df["coef"].values, color=colors, alpha=0.85,
            edgecolor=BG_BLACK)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["feature_clean"].values, fontsize=8.5)
    ax.set_xlabel(r"Coefficient ($\beta$, log-odds)")
    ax.set_title("Figure 15. Model Coefficients (Log-Odds Scale)", color=TEXT_WHITE)
    ax.axvline(0, color=TEXT_DIM, linewidth=0.8)

    risk_patch = mpatches.Patch(color=NEON_ORANGE, alpha=0.85,
                                 label="Risk factor (increases CKD odds)")
    protect_patch = mpatches.Patch(color=NEON_BLUE, alpha=0.85,
                                    label="Protective factor (decreases CKD odds)")
    ax.legend(handles=[risk_patch, protect_patch], loc="lower right", fontsize=9)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig15_coefficients.png", facecolor=BG_BLACK)
    plt.close()
    print("  -> fig15_coefficients.png")


def fig16_odds_ratio(coef_df):
    df = coef_df.copy()
    df["feature_clean"] = df["feature"].str.replace("missingindicator_", "MISSING: ")
    df["feature_clean"] = df["feature_clean"].str.replace("_", " ").str.title()
    df = df.sort_values("odds_ratio")

    fig, ax = plt.subplots(figsize=(10, 9))
    colors = [NEON_ORANGE if o > 1 else NEON_BLUE for o in df["odds_ratio"]]
    ax.scatter(df["odds_ratio"], range(len(df)), c=colors, s=70, zorder=3,
              edgecolors=BG_BLACK, linewidths=0.8)
    ax.hlines(range(len(df)), xmin=1, xmax=df["odds_ratio"],
              colors=colors, linewidth=1.5, alpha=0.7)
    ax.axvline(1, color=NEON_YELLOW, linewidth=1.2, linestyle="-", alpha=0.7)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["feature_clean"].values, fontsize=8.5)
    ax.set_xlabel("Odds Ratio (OR)")
    ax.set_title("Figure 16. Odds Ratios for CKD Risk", color=TEXT_WHITE)
    ax.set_xscale("log")
    ax.set_xlim(0.15, 5)

    risk_patch = mpatches.Patch(color=NEON_ORANGE, label="OR > 1 (risk factor)")
    protect_patch = mpatches.Patch(color=NEON_BLUE, label="OR < 1 (protective)")
    ax.legend(handles=[risk_patch, protect_patch], loc="lower right", fontsize=9)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig16_odds_ratios.png", facecolor=BG_BLACK)
    plt.close()
    print("  -> fig16_odds_ratios.png")


def main():
    print("=" * 60)
    print("RenAI v1 — DARK THEME Whitepaper Figures")
    print("=" * 60)

    print("\n[1/2] Running model pipeline...")
    df_valid, X_df, y, y_prob, coef_df = get_model_outputs()
    print(f"  Model trained: {len(y)} samples, {X_df.shape[1]} features")

    print("\n[2/2] Generating dark-theme figures...")
    fig6_missingness_bar(df_valid, y)
    fig7_sigmoid()
    fig8_egfr_curves()
    fig9_log_transform(df_valid)
    fig12_roc(y, y_prob)
    fig13_precision_recall(y, y_prob)
    fig14_calibration(y, y_prob)
    fig15_coefficient_bar(coef_df)
    fig16_odds_ratio(coef_df)

    print(f"\nAll figures saved to: {OUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()