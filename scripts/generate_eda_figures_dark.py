"""
Generate dark-theme EDA visualizations for the RenAI v1 whitepaper.
Black background, neon accent palette: blue, purple, yellow, orange.
Output: whitepaper/eda_visuals/
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.pipeline.run_pipeline import run_pipeline
from src.features.build_features import get_y

OUT_DIR = REPO / "whitepaper" / "eda_visuals"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Dark neon palette ─────────────────────────────────────────────────
NEON_BLUE = "#00D4FF"
NEON_PURPLE = "#B44CFF"
NEON_YELLOW = "#FFE600"
NEON_ORANGE = "#FF8C00"

CKD_COLOR = NEON_BLUE
CKD_COLOR_DIM = "#007A99"
NOTCKD_COLOR = NEON_ORANGE
NOTCKD_COLOR_DIM = "#994F00"

BG_BLACK = "#000000"
AXES_BG = "#0A0A0A"
GRID_COLOR = "#1A1A2E"
TEXT_WHITE = "#FFFFFF"
TEXT_DIM = "#888899"
EDGE_COLOR = "#222233"

plt.rcParams.update({
    "figure.facecolor": BG_BLACK,
    "axes.facecolor": AXES_BG,
    "axes.edgecolor": EDGE_COLOR,
    "axes.grid": True,
    "grid.color": GRID_COLOR,
    "grid.alpha": 0.6,
    "grid.linewidth": 0.4,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.labelcolor": TEXT_WHITE,
    "xtick.color": TEXT_WHITE,
    "ytick.color": TEXT_WHITE,
    "text.color": TEXT_WHITE,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "savefig.facecolor": BG_BLACK,
    "legend.facecolor": "#111122",
    "legend.edgecolor": EDGE_COLOR,
    "legend.labelcolor": TEXT_WHITE,
})

# ── Feature metadata ──────────────────────────────────────────────────
FEATURES = {
    "age": {"label": "Age", "unit": "years", "bins": 12},
    "blood_pressure": {"label": "Blood Pressure", "unit": "mmHg", "bins": 12},
    "specific_gravity": {"label": "Specific Gravity", "unit": "", "bins": 8},
    "albumin": {"label": "Albumin (Urine)", "unit": "scale 0–5", "bins": 6},
    "sugar": {"label": "Sugar (Urine)", "unit": "scale 0–5", "bins": 6},
    "blood_glucose_random": {"label": "Blood Glucose Random", "unit": "mg/dL", "bins": 15},
    "blood_urea": {"label": "Blood Urea", "unit": "mg/dL", "bins": 15},
    "serum_creatinine": {"label": "Serum Creatinine", "unit": "mg/dL", "bins": 15},
    "sodium": {"label": "Sodium", "unit": "mEq/L", "bins": 12},
    "potassium": {"label": "Potassium", "unit": "mEq/L", "bins": 12},
    "hemoglobin": {"label": "Hemoglobin", "unit": "g/dL", "bins": 12},
    "packed_cell_volume": {"label": "Packed Cell Volume", "unit": "%", "bins": 12},
    "white_blood_cell_count": {"label": "White Blood Cell Count", "unit": "cells/µL", "bins": 12},
    "red_blood_cell_count": {"label": "Red Blood Cell Count", "unit": "M/µL", "bins": 10},
    "egfr_cr": {"label": "eGFR (Creatinine)", "unit": "mL/min/1.73m²", "bins": 15},
    "urea_creatinine_ratio": {"label": "Urea / Creatinine Ratio", "unit": "", "bins": 12},
}


def compute_monotonicity(vals, labels, n_bins=8):
    mask = np.isfinite(vals) & np.isfinite(labels)
    v, l = vals[mask], labels[mask]
    if len(v) < 20:
        return np.array([]), np.array([]), np.array([])
    try:
        bin_edges = np.percentile(v, np.linspace(0, 100, n_bins + 1))
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 3:
            bin_edges = np.linspace(v.min(), v.max(), n_bins + 1)
    except Exception:
        bin_edges = np.linspace(v.min(), v.max(), n_bins + 1)

    centers, probs, counts = [], [], []
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (v >= lo) & (v <= hi) if i == len(bin_edges) - 2 else (v >= lo) & (v < hi)
        if in_bin.sum() >= 5:
            centers.append((lo + hi) / 2)
            probs.append(l[in_bin].mean())
            counts.append(in_bin.sum())
    return np.array(centers), np.array(probs), np.array(counts)


def make_feature_figure(col, ckd_vals, notckd_vals, all_vals, labels, meta, fig_num):
    label = meta["label"]
    unit = meta["unit"]
    n_bins = meta["bins"]
    xlabel = f"{label} ({unit})" if unit else label

    fig = plt.figure(figsize=(16, 4.8))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.1, 0.8, 1.0], wspace=0.32)

    # ── Panel A: Histograms + KDE ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    all_finite = all_vals[np.isfinite(all_vals)]
    if len(all_finite) < 10:
        plt.close()
        return None
    bin_lo, bin_hi = np.percentile(all_finite, [1, 99])
    bins = np.linspace(bin_lo, bin_hi, n_bins + 1)

    ckd_f = ckd_vals[(ckd_vals >= bin_lo) & (ckd_vals <= bin_hi)]
    notckd_f = notckd_vals[(notckd_vals >= bin_lo) & (notckd_vals <= bin_hi)]

    ax1.hist(ckd_f, bins=bins, density=True, alpha=0.50, color=CKD_COLOR,
             edgecolor=BG_BLACK, linewidth=0.6, label="CKD", zorder=2)
    ax1.hist(notckd_f, bins=bins, density=True, alpha=0.45, color=NOTCKD_COLOR,
             edgecolor=BG_BLACK, linewidth=0.6, label="non-CKD", zorder=2)

    try:
        if len(ckd_f) > 5 and ckd_f.std() > 0:
            kde_ckd = stats.gaussian_kde(ckd_f, bw_method=0.3)
            x_kde = np.linspace(bin_lo, bin_hi, 200)
            ax1.plot(x_kde, kde_ckd(x_kde), color=CKD_COLOR, linewidth=2.5, zorder=3)
        if len(notckd_f) > 5 and notckd_f.std() > 0:
            kde_notckd = stats.gaussian_kde(notckd_f, bw_method=0.3)
            x_kde = np.linspace(bin_lo, bin_hi, 200)
            ax1.plot(x_kde, kde_notckd(x_kde), color=NOTCKD_COLOR, linewidth=2.5, zorder=3)
    except Exception:
        pass

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Density")
    ax1.set_title("A)  Distribution (CKD vs non-CKD)", color=TEXT_WHITE)
    ax1.legend(frameon=True, fancybox=True, shadow=False, fontsize=9)
    ax1.set_xlim(bin_lo, bin_hi)

    # ── Panel B: Box Plot ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    bp = ax2.boxplot(
        [ckd_f, notckd_f], positions=[1, 2], widths=0.5, patch_artist=True,
        showfliers=True,
        flierprops=dict(marker="o", markersize=3, alpha=0.5, markerfacecolor=TEXT_DIM,
                        markeredgecolor="none"),
        medianprops=dict(color=NEON_YELLOW, linewidth=2.5),
        whiskerprops=dict(color=TEXT_DIM, linewidth=1.2),
        capprops=dict(color=TEXT_DIM, linewidth=1.2),
    )
    bp["boxes"][0].set_facecolor(CKD_COLOR_DIM)
    bp["boxes"][0].set_alpha(0.8)
    bp["boxes"][0].set_edgecolor(CKD_COLOR)
    bp["boxes"][1].set_facecolor(NOTCKD_COLOR_DIM)
    bp["boxes"][1].set_alpha(0.8)
    bp["boxes"][1].set_edgecolor(NOTCKD_COLOR)

    for i, (data, color) in enumerate([(ckd_f, CKD_COLOR), (notckd_f, NOTCKD_COLOR)]):
        jitter = np.random.default_rng(42).normal(0, 0.06, len(data))
        ax2.scatter(np.full_like(data, i + 1) + jitter, data,
                    c=color, s=6, alpha=0.25, zorder=1, edgecolors="none")

    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(["CKD", "non-CKD"], fontweight="bold")
    ax2.set_ylabel(xlabel)
    ax2.set_title("B)  Box Plot by Class", color=TEXT_WHITE)

    for i, d in enumerate([ckd_f, notckd_f]):
        med = np.median(d)
        ax2.annotate(f"{med:.1f}", xy=(i + 1, med), xytext=(i + 1.32, med),
                     fontsize=8, color=TEXT_WHITE, va="center",
                     arrowprops=dict(arrowstyle="-", color=TEXT_DIM, linewidth=0.5))

    # ── Panel C: Monotonicity ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    centers, probs, counts = compute_monotonicity(all_vals, labels, n_bins=10)

    if len(centers) > 1:
        ax3.fill_between(centers, probs, alpha=0.15, color=NEON_PURPLE, zorder=1)
        ax3.plot(centers, probs, "o-", color=NEON_PURPLE, linewidth=2.5,
                 markersize=7, markeredgecolor=BG_BLACK, markeredgewidth=1.2, zorder=3)
        sizes = (counts / counts.max()) * 120 + 20
        ax3.scatter(centers, probs, s=sizes, color=NEON_PURPLE, alpha=0.15, zorder=2)
        ax3.axhline(0.5, color=TEXT_DIM, linestyle="--", linewidth=0.8, alpha=0.6)
        ax3.annotate("P = 0.5", xy=(centers[-1], 0.5), fontsize=7,
                     color=TEXT_DIM, va="bottom", ha="right")

    ax3.set_xlabel(xlabel)
    ax3.set_ylabel("P(CKD)")
    ax3.set_title("C)  CKD Probability vs Feature Value", color=TEXT_WHITE)
    ax3.set_ylim(-0.05, 1.05)
    ax3.yaxis.set_major_locator(MaxNLocator(6))

    fig.suptitle(f"Figure {fig_num}.  {label}", fontsize=14, fontweight="bold",
                 y=1.02, color=TEXT_WHITE)

    fname = f"fig_{fig_num:02d}_{col}.png"
    fig.savefig(OUT_DIR / fname, facecolor=BG_BLACK)
    plt.close()
    return fname


def make_correlation_heatmap(df_valid, y, features_list):
    cols = [c for c in features_list if c in df_valid.columns]
    data = df_valid[cols].apply(pd.to_numeric, errors="coerce")
    data["CKD"] = y
    corr = data.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    labels = [c.replace("_", " ").title() for c in corr.columns]
    n = len(labels)

    # Custom purple-blue colormap
    from matplotlib.colors import LinearSegmentedColormap
    dark_cmap = LinearSegmentedColormap.from_list("neon_div",
        [NEON_ORANGE, "#1A1A1A", NEON_BLUE], N=256)

    im = ax.imshow(np.where(mask, np.nan, corr.values), cmap=dark_cmap,
                   vmin=-1, vmax=1, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    for i in range(n):
        for j in range(n):
            if not mask[i, j]:
                val = corr.values[i, j]
                color = BG_BLACK if abs(val) > 0.6 else TEXT_WHITE
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6.5, color=color)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
    cbar.set_label("Pearson Correlation", fontsize=10, color=TEXT_WHITE)
    cbar.ax.yaxis.set_tick_params(color=TEXT_WHITE)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=TEXT_WHITE)

    ax.set_title("Feature Correlation Matrix (with CKD Target)", fontsize=14,
                 fontweight="bold", pad=15, color=TEXT_WHITE)
    fig.savefig(OUT_DIR / "fig_correlation_heatmap.png", facecolor=BG_BLACK)
    plt.close()
    print("  -> fig_correlation_heatmap.png")


def make_class_balance_figure(y):
    n_ckd = int((y == 1).sum())
    n_notckd = int((y == 0).sum())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5),
                                    gridspec_kw={"width_ratios": [1, 1.2]})

    sizes = [n_ckd, n_notckd]
    colors = [CKD_COLOR, NOTCKD_COLOR]
    wedges, texts, autotexts = ax1.pie(
        sizes, colors=colors, autopct="%1.1f%%", startangle=90, pctdistance=0.75,
        textprops=dict(fontsize=12, fontweight="bold", color=BG_BLACK),
        wedgeprops=dict(width=0.45, edgecolor=BG_BLACK, linewidth=2)
    )
    centre = plt.Circle((0, 0), 0.55, fc=BG_BLACK)
    ax1.add_artist(centre)
    ax1.text(0, 0, f"N={len(y)}", ha="center", va="center", fontsize=14,
             fontweight="bold", color=TEXT_WHITE)
    ax1.legend(["CKD", "non-CKD"], loc="lower center", fontsize=10,
               frameon=False, ncol=2, bbox_to_anchor=(0.5, -0.08))
    ax1.set_title("A)  Class Distribution", fontweight="bold", color=TEXT_WHITE)

    bars = ax2.bar(["CKD", "non-CKD"], [n_ckd, n_notckd],
                   color=colors, alpha=0.85, edgecolor=BG_BLACK, linewidth=1.5, width=0.5)
    for bar, count in zip(bars, [n_ckd, n_notckd]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                 str(count), ha="center", va="bottom", fontsize=13,
                 fontweight="bold", color=TEXT_WHITE)
    ax2.set_ylabel("Number of Patients")
    ax2.set_title("B)  Class Counts", fontweight="bold", color=TEXT_WHITE)
    ax2.set_ylim(0, max(n_ckd, n_notckd) * 1.15)

    fig.suptitle("Dataset Class Balance", fontsize=14, fontweight="bold",
                 y=1.02, color=TEXT_WHITE)
    fig.savefig(OUT_DIR / "fig_class_balance.png", facecolor=BG_BLACK)
    plt.close()
    print("  -> fig_class_balance.png")


def make_missingness_heatmap(df_valid, y):
    features = [c for c in FEATURES if c in df_valid.columns]
    data = df_valid[features].apply(pd.to_numeric, errors="coerce")

    miss_overall = data.isna().mean() * 100
    miss_ckd = data[y == 1].isna().mean() * 100
    miss_notckd = data[y == 0].isna().mean() * 100

    miss_df = pd.DataFrame({
        "Feature": [FEATURES[f]["label"] for f in features],
        "Overall": [miss_overall[f] for f in features],
        "CKD": [miss_ckd[f] for f in features],
        "non-CKD": [miss_notckd[f] for f in features],
    }).sort_values("Overall", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = np.arange(len(miss_df))
    h = 0.28

    ax.barh(y_pos + h, miss_df["CKD"], h, color=NEON_BLUE, alpha=0.85,
            label="CKD", edgecolor=BG_BLACK, linewidth=0.5)
    ax.barh(y_pos, miss_df["non-CKD"], h, color=NEON_ORANGE, alpha=0.85,
            label="non-CKD", edgecolor=BG_BLACK, linewidth=0.5)
    ax.barh(y_pos - h, miss_df["Overall"], h, color=NEON_PURPLE, alpha=0.5,
            label="Overall", edgecolor=BG_BLACK, linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(miss_df["Feature"], fontsize=9)
    ax.set_xlabel("Missing (%)", fontsize=11)
    ax.set_title("Missingness Rate by Feature and CKD Status", fontsize=14,
                 fontweight="bold", color=TEXT_WHITE)
    ax.legend(loc="lower right", frameon=True, fancybox=True)

    for i, (_, row) in enumerate(miss_df.iterrows()):
        if row["CKD"] > 5 and row["non-CKD"] > 0:
            ratio = row["CKD"] / row["non-CKD"]
            ax.annotate(f"{ratio:.1f}x", xy=(row["CKD"] + 0.8, y_pos[i] + h),
                        fontsize=7, color=TEXT_WHITE, va="center")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig_missingness_detailed.png", facecolor=BG_BLACK)
    plt.close()
    print("  -> fig_missingness_detailed.png")


def main():
    print("=" * 60)
    print("RenAI v1 — DARK THEME EDA Visualizations")
    print("=" * 60)

    print("\n[1/3] Loading data...")
    pipe_out = run_pipeline("data/raw/CKD.csv", require_target=True)
    df_valid = pipe_out.df_valid
    y = get_y(df_valid, require_target=True)
    print(f"  {len(y)} patients loaded ({(y == 1).sum()} CKD, {(y == 0).sum()} non-CKD)")

    print("\n[2/3] Generating per-feature EDA panels...")
    fig_num = 1
    features_order = [
        "egfr_cr", "serum_creatinine", "hemoglobin", "albumin",
        "specific_gravity", "packed_cell_volume", "blood_urea",
        "red_blood_cell_count", "blood_glucose_random", "age",
        "blood_pressure", "sodium", "potassium",
        "white_blood_cell_count", "urea_creatinine_ratio", "sugar",
    ]

    for col in features_order:
        if col not in df_valid.columns or col not in FEATURES:
            continue
        meta = FEATURES[col]
        vals_raw = pd.to_numeric(df_valid[col], errors="coerce").values
        valid_mask = np.isfinite(vals_raw)
        ckd_vals = vals_raw[(y == 1) & valid_mask]
        notckd_vals = vals_raw[(y == 0) & valid_mask]
        all_vals = vals_raw.copy()
        all_labels = y.astype(float).copy()

        fname = make_feature_figure(col, ckd_vals, notckd_vals, all_vals,
                                     all_labels, meta, fig_num)
        if fname:
            print(f"  -> {fname}")
            fig_num += 1

    print("\n[3/3] Generating summary figures...")
    make_class_balance_figure(y)
    make_correlation_heatmap(df_valid, y, list(FEATURES.keys()))
    make_missingness_heatmap(df_valid, y)

    print(f"\nAll {fig_num - 1 + 3} figures saved to: {OUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()