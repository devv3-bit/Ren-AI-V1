"""
Step 11: figures for Ren AI v2 (dark theme matching
scripts/generate_whitepaper_figures_dark.py).

Reads ONLY the saved outputs of scripts 04 / 05 / 06 / 07; it never
evaluates a model. Output: reports/nhanes/figures/*.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.nhanes.constants import FIGURES_DIR, REPORTS_DIR, TARGET_COL, EARLY_SUBGROUP_COL  # noqa: E402

# -- Dark neon palette (same as the v1 whitepaper figures) ----------------------
NEON_BLUE = "#00D4FF"
NEON_PURPLE = "#B44CFF"
NEON_YELLOW = "#FFE600"
NEON_ORANGE = "#FF8C00"
NEON_PINK = "#FF5C8A"
BG_BLACK = "#000000"
AXES_BG = "#0A0A0A"
GRID_COLOR = "#1A1A2E"
TEXT_WHITE = "#FFFFFF"
TEXT_DIM = "#888899"
EDGE_COLOR = "#222233"

plt.rcParams.update({
    "figure.facecolor": BG_BLACK, "axes.facecolor": AXES_BG, "axes.edgecolor": EDGE_COLOR,
    "axes.grid": True, "grid.color": GRID_COLOR, "grid.alpha": 0.5, "grid.linewidth": 0.4,
    "font.family": "sans-serif", "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
    "axes.labelcolor": TEXT_WHITE, "axes.titlecolor": TEXT_WHITE, "xtick.color": TEXT_WHITE,
    "ytick.color": TEXT_WHITE, "text.color": TEXT_WHITE, "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.facecolor": BG_BLACK, "savefig.bbox": "tight", "legend.facecolor": "#111122",
    "legend.edgecolor": EDGE_COLOR, "legend.labelcolor": TEXT_WHITE,
})

MODEL_STYLE = {
    "model_B_full": ("Model B: gradient boosting, 25 features", NEON_BLUE),
    "model_A_full": ("Model A: logistic regression, 25 features", NEON_PURPLE),
    "model_A_harmonized": ("Model A, 14 shared features (UCI-compatible)", NEON_PINK),
    "baseline_egfr": ("Baseline: eGFR alone", NEON_YELLOW),
    "baseline_demographic": ("Baseline: age + sex + diabetes + BP", NEON_ORANGE),
}
PRETTY = {
    "egfr_cr": "eGFR (CKD-EPI 2021)", "bun_creatinine_ratio": "BUN : creatinine ratio", "bun": "BUN",
    "hba1c": "HbA1c", "sbp": "Systolic BP", "blood_pressure": "Diastolic BP", "bmi": "BMI",
    "sex_male": "Male sex", "diabetes_mellitus": "Diabetes (self-reported)", "serum_creatinine": "Serum creatinine",
    "blood_glucose_random": "Glucose (random)", "serum_albumin": "Serum albumin", "uric_acid": "Uric acid",
    "packed_cell_volume": "Hematocrit (PCV)", "red_blood_cell_count": "RBC count",
    "white_blood_cell_count": "WBC count", "age": "Age",
}


def pretty(name: str) -> str:
    if name.startswith("missingindicator_"):
        return "MISSING: " + pretty(name[len("missingindicator_"):])
    return PRETTY.get(name, name.replace("_", " ").title())


def _auc_label(block: dict) -> str:
    return f"AUC {block['auc']:.3f} [{block['ci_low']:.3f}-{block['ci_high']:.3f}]"


def _save(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / name, facecolor=BG_BLACK)
    plt.close(fig)
    print(f"  -> {name}")


# -- Figure 1: cohort flow ------------------------------------------------------
EXCLUSION_LABELS = {1: "age < 18 years", 2: "pregnant at exam", 3: "creatinine or ACR not measured"}


def fig01_cohort_flow(summary: dict) -> None:
    steps = summary["flow"]
    by_split = summary["by_split"]
    fig, ax = plt.subplots(figsize=(10, 8.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.grid(False)

    def box(x, y, w, h, text, color, fontsize=10, weight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    linewidth=1.8, edgecolor=color, facecolor="#0E0E1A"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
                color=TEXT_WHITE, fontweight=weight, linespacing=1.5)

    ys = [8.6, 6.5, 4.4, 2.3]
    labels = [
        f"All participants, 7 NHANES cycles\n2005-2016 and 2017-Mar 2020\nn = {steps[0]['n_remaining']:,}",
        f"Age >= 18 years\nn = {steps[1]['n_remaining']:,}",
        f"Not pregnant at exam\nn = {steps[2]['n_remaining']:,}",
        f"Serum creatinine AND urine ACR measured\n(both required for the KDIGO label)\nn = {steps[3]['n_remaining']:,}",
    ]
    for i, (y, lab) in enumerate(zip(ys, labels)):
        box(0.8, y, 5.2, 1.25, lab, NEON_BLUE if i < 3 else NEON_PURPLE, weight="bold" if i == 3 else "normal")
        if i < 3:
            ax.annotate("", xy=(3.4, ys[i + 1] + 1.25), xytext=(3.4, y),
                        arrowprops=dict(arrowstyle="-|>", color=TEXT_WHITE, linewidth=1.5))
            excl = steps[i + 1]
            box(6.6, y - 0.95, 3.0, 0.8, f"Excluded: {excl['n_excluded']:,}\n{EXCLUSION_LABELS[i + 1]}",
                NEON_ORANGE, fontsize=8.5)
            ax.annotate("", xy=(6.6, y - 0.55), xytext=(3.4, y - 0.55),
                        arrowprops=dict(arrowstyle="-|>", color=NEON_ORANGE, linewidth=1.2))
    train, test = by_split["train"], by_split["test"]
    ax.annotate("", xy=(2.0, 1.5), xytext=(2.5, 2.3), arrowprops=dict(arrowstyle="-|>", color=TEXT_WHITE, linewidth=1.3))
    ax.annotate("", xy=(5.0, 1.5), xytext=(4.3, 2.3), arrowprops=dict(arrowstyle="-|>", color=TEXT_WHITE, linewidth=1.3))
    box(0.4, 0.35, 3.3, 1.1, f"TRAIN: cycles 2005-2016\nn = {train['n']:,}  (CKD {100 * train['prevalence']:.1f}%)", NEON_YELLOW, fontsize=9.5)
    box(3.9, 0.35, 3.3, 1.1, f"TEST: 2017-Mar 2020 (held out)\nn = {test['n']:,}  (CKD {100 * test['prevalence']:.1f}%)", NEON_PINK, fontsize=9.5)
    ax.set_title("Figure 1. NHANES cohort flow and temporal split", color=TEXT_WHITE, fontsize=13)
    _save(fig, "fig01_cohort_flow.png")


# -- Figure 2: ROC on the held-out NHANES cycle --------------------------------
def fig02_roc_test(preds: pd.DataFrame, metrics: dict) -> None:
    final = metrics["final_model"]
    order = [final] + [m for m in MODEL_STYLE if m != final and m in metrics["models"]]
    y_all = preds[TARGET_COL].to_numpy().astype(int)
    early = preds[EARLY_SUBGROUP_COL].to_numpy().astype(bool)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.3))
    panels = [
        (axes[0], np.ones_like(early, dtype=bool), "roc_auc",
         f"A) All test participants (n = {metrics['n_test']:,}, CKD {100 * metrics['prevalence']:.1f}%)"),
        (axes[1], early, "early",
         f"B) Early subgroup, eGFR >= 60 (n = {metrics['n_early_subgroup']:,}, CKD {100 * metrics['prevalence_early_subgroup']:.1f}%)"),
    ]
    for ax, mask, key, title in panels:
        for name in order:
            label, color = MODEL_STYLE[name]
            prob = preds[f"prob_{name}"].to_numpy()
            fpr, tpr, _ = roc_curve(y_all[mask], prob[mask])
            block = metrics["models"][name]["roc_auc"] if key == "roc_auc" else metrics["models"][name]["early_subgroup"]["roc_auc"]
            lw = 2.8 if name == final else 1.7
            ax.plot(fpr, tpr, color=color, linewidth=lw, alpha=1.0 if name == final else 0.9,
                    label=f"{label}\n   {_auc_label(block)}")
            if name == final:
                ax.fill_between(fpr, tpr, alpha=0.08, color=color)
        ax.plot([0, 1], [0, 1], color=TEXT_DIM, linestyle="--", linewidth=1, label="Random (AUC 0.500)")
        ax.set_xlabel("False positive rate (1 - specificity)")
        ax.set_ylabel("True positive rate (sensitivity)")
        ax.set_title(title, color=TEXT_WHITE, fontsize=11)
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.set_aspect("equal")
        ax.legend(loc="lower right", fontsize=7.6)
    fig.suptitle("Figure 2. ROC on held-out NHANES 2017-Mar 2020 (models frozen on 2005-2016; 95% bootstrap CIs)",
                 color=TEXT_WHITE, fontsize=12.5, y=0.99)
    plt.tight_layout()
    _save(fig, "fig02_roc_nhanes_test.png")


# -- Figure 3: ROC on the UCI hospital patients --------------------------------
def fig03_roc_uci(uci_preds: pd.DataFrame, ext: dict) -> None:
    y = uci_preds["ckd"].to_numpy().astype(int)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for sex, color, lw in (("male", NEON_PINK, 2.6), ("female", NEON_PURPLE, 1.6)):
        prob = uci_preds[f"prob_egfr_{sex}"].to_numpy()
        fpr, tpr, _ = roc_curve(y, prob)
        block = ext["by_sex_assumption"][sex]["roc_auc"]
        tag = "primary" if sex == "male" else "sensitivity"
        ax.plot(fpr, tpr, color=color, linewidth=lw, label=f"eGFR computed as {sex} ({tag})\n   {_auc_label(block)}")
        if sex == "male":
            ax.fill_between(fpr, tpr, alpha=0.1, color=color)
    ax.plot([0, 1], [0, 1], color=TEXT_DIM, linestyle="--", linewidth=1, label="Random (AUC 0.500)")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title(f"Figure 3. External validation: UCI hospital patients (n = {ext['n_uci']})\n"
                 f"Model A (14 shared features) trained on NHANES 2005-2016, applied unchanged",
                 color=TEXT_WHITE, fontsize=10.5)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=8.5)
    plt.tight_layout()
    _save(fig, "fig03_roc_uci_external.png")


# -- Figure 4: calibration on the held-out NHANES cycle ------------------------
def fig04_calibration(metrics: dict) -> None:
    final = metrics["final_model"]
    names = [final] + [n for n in ("model_A_full", "model_B_full") if n != final]
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for name, marker in zip(names, ("o", "s")):
        label, color = MODEL_STYLE[name]
        cal = pd.DataFrame(metrics["models"][name]["calibration"])
        ax.plot(cal["mean_pred"], cal["frac_pos"], marker=marker, linestyle="-", color=color, linewidth=2,
                markersize=7, markeredgecolor=BG_BLACK,
                label=f"{label}\n   Brier {metrics['models'][name]['brier']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color=TEXT_DIM, linewidth=1, label="Perfectly calibrated")
    ax.axhline(metrics["prevalence"], color=NEON_YELLOW, linestyle=":", linewidth=1,
               label=f"Test prevalence {metrics['prevalence']:.3f}")
    ax.set_xlabel("Mean predicted probability (decile bins)")
    ax.set_ylabel("Observed fraction with CKD")
    ax.set_title("Figure 4. Calibration on held-out NHANES 2017-Mar 2020", color=TEXT_WHITE)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    _save(fig, "fig04_calibration_nhanes_test.png")


# -- Figure 5: odds ratios of Model A (top 15 by |log-odds|) -------------------
def fig05_odds_ratios(coef: pd.DataFrame) -> None:
    df = coef.copy()
    df = df.reindex(df["coef_log_odds"].abs().sort_values(ascending=False).index).head(15)
    df = df.sort_values("odds_ratio")
    df["label"] = df["feature"].map(pretty)
    colors = [NEON_ORANGE if o > 1 else NEON_BLUE for o in df["odds_ratio"]]

    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.scatter(df["odds_ratio"], range(len(df)), c=colors, s=80, zorder=3, edgecolors=BG_BLACK, linewidths=0.8)
    ax.hlines(range(len(df)), xmin=1, xmax=df["odds_ratio"], colors=colors, linewidth=1.6, alpha=0.75)
    ax.axvline(1, color=NEON_YELLOW, linewidth=1.2, alpha=0.7)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["label"].values, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Odds ratio per 1 SD of the (log-)transformed feature; indicators per 1 unit")
    ax.set_title("Figure 5. Model A odds ratios on NHANES 2005-2016 (top 15 of the L2 logistic regression)",
                 color=TEXT_WHITE, fontsize=11.5)
    ax.legend(handles=[mpatches.Patch(color=NEON_ORANGE, label="OR > 1 (higher CKD odds)"),
                       mpatches.Patch(color=NEON_BLUE, label="OR < 1 (lower CKD odds)")],
              loc="lower right", fontsize=9)
    plt.tight_layout()
    _save(fig, "fig05_model_a_odds_ratios.png")


# -- Figure 6: permutation importance of the final model (top 15) --------------
def fig06_permutation_importance(pi: pd.DataFrame, final: str) -> None:
    df = pi.sort_values("importance_mean", ascending=False).head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.barh(range(len(df)), df["importance_mean"], xerr=df["importance_std"], color=NEON_BLUE, alpha=0.85,
            edgecolor=BG_BLACK, error_kw=dict(ecolor=TEXT_WHITE, capsize=3, linewidth=1))
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([pretty(f) for f in df["feature"]], fontsize=9)
    ax.set_xlabel("Drop in test ROC-AUC when the feature is permuted (mean of 5 repeats)")
    ax.set_title(f"Figure 6. Permutation importance of the final model ({MODEL_STYLE[final][0]})\n"
                 "computed on held-out NHANES 2017-Mar 2020, top 15", color=TEXT_WHITE, fontsize=11)
    plt.tight_layout()
    _save(fig, "fig06_permutation_importance.png")


# -- Figure 7: ROC on UCI, v2 first look vs v3 second look ----------------------
def fig07_roc_uci_v3(uci_preds_v2: pd.DataFrame, ext_v2: dict, uci_preds_v3: pd.DataFrame, ext_v3: dict) -> None:
    y = uci_preds_v2["ckd"].to_numpy().astype(int)
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    fpr, tpr, _ = roc_curve(y, uci_preds_v2["prob_egfr_male"].to_numpy())
    ax.plot(fpr, tpr, color=TEXT_DIM, linewidth=1.6, linestyle="-.",
            label=f"v2 harmonised logistic, first look\n   {_auc_label(ext_v2['by_sex_assumption']['male']['roc_auc'])}")
    styles = {"transfer_hgb_14": ("v3 gradient boosting, 14 features", NEON_BLUE),
              "transfer_lr_v3": ("v3 logistic, no indicators + clipping", NEON_PINK),
              "transfer_lr_v3_noclip": ("ablation: no indicators, no clipping", NEON_YELLOW)}
    final = ext_v3["final_model"]
    for name in [final] + [n for n in styles if n != final]:
        if f"prob_{name}" not in uci_preds_v3.columns:
            continue
        label, color = styles[name]
        fpr, tpr, _ = roc_curve(y, uci_preds_v3[f"prob_{name}"].to_numpy())
        block = ext_v3["models"][name]["uci"]["male"]["roc_auc"]
        ax.plot(fpr, tpr, color=color, linewidth=2.8 if name == final else 1.6,
                label=f"{label}{' (FINAL v3)' if name == final else ''}\n   {_auc_label(block)}")
        if name == final:
            ax.fill_between(fpr, tpr, alpha=0.1, color=color)
    ax.plot([0, 1], [0, 1], color=TEXT_DIM, linestyle="--", linewidth=1, label="Random (AUC 0.500)")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title(f"Figure 7. UCI hospital patients (n = {ext_v3['n_uci']}): v3 transfer models\n"
                 "(selected by NHANES CV only; disclosed second look at UCI)", color=TEXT_WHITE, fontsize=10.5)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    _save(fig, "fig07_roc_uci_v3.png")


def main() -> None:
    print("Ren AI v2 - NHANES figures (dark theme)")
    summary = json.loads((REPORTS_DIR / "cohort_summary.json").read_text())
    metrics = json.loads((REPORTS_DIR / "test_metrics.json").read_text())
    ext = json.loads((REPORTS_DIR / "external_metrics.json").read_text())
    preds = pd.read_csv(REPORTS_DIR / "test_predictions.csv")
    uci_preds = pd.read_csv(REPORTS_DIR / "uci_predictions.csv")
    coef = pd.read_csv(REPORTS_DIR / "model_A_coefficients.csv")
    pi = pd.read_csv(REPORTS_DIR / "permutation_importance.csv")

    fig01_cohort_flow(summary)
    fig02_roc_test(preds, metrics)
    fig03_roc_uci(uci_preds, ext)
    fig04_calibration(metrics)
    fig05_odds_ratios(coef)
    fig06_permutation_importance(pi, metrics["final_model"])
    v3_path = REPORTS_DIR / "external_metrics_v3.json"
    if v3_path.exists():
        ext_v3 = json.loads(v3_path.read_text())
        uci_preds_v3 = pd.read_csv(REPORTS_DIR / "uci_predictions_v3.csv")
        fig07_roc_uci_v3(uci_preds, ext, uci_preds_v3, ext_v3)
    print(f"all figures saved to {FIGURES_DIR.relative_to(REPO)}")


if __name__ == "__main__":
    main()
