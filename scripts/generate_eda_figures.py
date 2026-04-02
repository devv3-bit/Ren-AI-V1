"""
Generate polished EDA visualizations for the RenAI v1 whitepaper.
Each feature gets a 3-panel figure:
  A) Overlapping histograms with KDE (CKD vs non-CKD)
  B) Split violin plot by class
  C) Monotonicity curve: P(CKD) vs binned feature
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

# ── Style ──────────────────────────────────────────────────────────────
CKD_BLUE = "#1a6fb5"
CKD_BLUE_LIGHT = "#a8d4f2"
NOTCKD_CORAL = "#e8734a"
NOTCKD_CORAL_LIGHT = "#f9c9b3"
BG_COLOR = "#fafafa"
GRID_COLOR = "#e0e0e0"
TEXT_COLOR = "#2d2d2d"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": BG_COLOR,
    "axes.edgecolor": "#cccccc",
    "axes.grid": True,
    "grid.color": GRID_COLOR,
    "grid.alpha": 0.5,
    "grid.linewidth": 0.5,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "text.color": TEXT_COLOR,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
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
    """Bin feature values and compute P(CKD) per bin."""
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
        if i == len(bin_edges) - 2:
            in_bin = (v >= lo) & (v <= hi)
        else:
            in_bin = (v >= lo) & (v < hi)
        if in_bin.sum() >= 5:
            centers.append((lo + hi) / 2)
            probs.append(l[in_bin].mean())
            counts.append(in_bin.sum())

    return np.array(centers), np.array(probs), np.array(counts)


def make_feature_figure(col, ckd_vals, notckd_vals, all_vals, labels, meta, fig_num):
    """Create a polished 3-panel figure for one feature."""
    label = meta["label"]
    unit = meta["unit"]
    n_bins = meta["bins"]
    xlabel = f"{label} ({unit})" if unit else label

    fig = plt.figure(figsize=(16, 4.8))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.1, 0.8, 1.0], wspace=0.32)

    # ── Panel A: Overlapping Histograms + KDE ──────────────────────────
    ax1 = fig.add_subplot(gs[0])

    # Compute shared bins
    all_finite = all_vals[np.isfinite(all_vals)]
    if len(all_finite) < 10:
        return
    bin_lo, bin_hi = np.percentile(all_finite, [1, 99])
    bins = np.linspace(bin_lo, bin_hi, n_bins + 1)

    ckd_f = ckd_vals[(ckd_vals >= bin_lo) & (ckd_vals <= bin_hi)]
    notckd_f = notckd_vals[(notckd_vals >= bin_lo) & (notckd_vals <= bin_hi)]

    ax1.hist(ckd_f, bins=bins, density=True, alpha=0.45, color=CKD_BLUE,
             edgecolor="white", linewidth=0.6, label="CKD", zorder=2)
    ax1.hist(notckd_f, bins=bins, density=True, alpha=0.40, color=NOTCKD_CORAL,
             edgecolor="white", linewidth=0.6, label="non-CKD", zorder=2)

    # KDE curves
    try:
        if len(ckd_f) > 5 and ckd_f.std() > 0:
            kde_ckd = stats.gaussian_kde(ckd_f, bw_method=0.3)
            x_kde = np.linspace(bin_lo, bin_hi, 200)
            ax1.plot(x_kde, kde_ckd(x_kde), color=CKD_BLUE, linewidth=2.2, zorder=3)
        if len(notckd_f) > 5 and notckd_f.std() > 0:
            kde_notckd = stats.gaussian_kde(notckd_f, bw_method=0.3)
            x_kde = np.linspace(bin_lo, bin_hi, 200)
            ax1.plot(x_kde, kde_notckd(x_kde), color=NOTCKD_CORAL, linewidth=2.2, zorder=3)
    except Exception:
        pass

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Density")
    ax1.set_title("A)  Distribution (CKD vs non-CKD)")
    ax1.legend(frameon=True, fancybox=True, shadow=False, framealpha=0.9,
               edgecolor="#dddddd", fontsize=9)
    ax1.set_xlim(bin_lo, bin_hi)

    # ── Panel B: Split Violin / Box Plot ───────────────────────────────
    ax2 = fig.add_subplot(gs[1])

    bp_data = [ckd_f, notckd_f]
    bp = ax2.boxplot(
        bp_data,
        positions=[1, 2],
        widths=0.5,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(color="#666666", linewidth=1.2),
        capprops=dict(color="#666666", linewidth=1.2),
    )
    bp["boxes"][0].set_facecolor(CKD_BLUE)
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor(NOTCKD_CORAL)
    bp["boxes"][1].set_alpha(0.7)
    for b in bp["boxes"]:
        b.set_edgecolor("#555555")
        b.set_linewidth(0.8)

    # Add individual points (jittered)
    for i, (data, color) in enumerate([(ckd_f, CKD_BLUE_LIGHT), (notckd_f, NOTCKD_CORAL_LIGHT)]):
        jitter = np.random.default_rng(42).normal(0, 0.06, len(data))
        ax2.scatter(np.full_like(data, i + 1) + jitter, data,
                    c=color, s=6, alpha=0.3, zorder=1, edgecolors="none")

    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(["CKD", "non-CKD"], fontweight="bold")
    ax2.set_ylabel(xlabel)
    ax2.set_title("B)  Box Plot by Class")

    # Add median labels
    for i, d in enumerate(bp_data):
        med = np.median(d)
        ax2.annotate(f"{med:.1f}", xy=(i + 1, med), xytext=(i + 1.32, med),
                     fontsize=8, color="#444444", va="center",
                     arrowprops=dict(arrowstyle="-", color="#aaaaaa", linewidth=0.5))

    # ── Panel C: Monotonicity ──────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])

    centers, probs, counts = compute_monotonicity(all_vals, labels, n_bins=10)

    if len(centers) > 1:
        # Area fill
        ax3.fill_between(centers, probs, alpha=0.12, color=CKD_BLUE, zorder=1)
        # Line
        ax3.plot(centers, probs, "o-", color=CKD_BLUE, linewidth=2.2,
                 markersize=7, markeredgecolor="white", markeredgewidth=1.2, zorder=3)
        # Point size by count
        sizes = (counts / counts.max()) * 120 + 20
        ax3.scatter(centers, probs, s=sizes, color=CKD_BLUE, alpha=0.15, zorder=2)

        # Reference line at 0.5
        ax3.axhline(0.5, color="#999999", linestyle="--", linewidth=0.8, alpha=0.6)
        ax3.annotate("P = 0.5", xy=(centers[-1], 0.5), fontsize=7,
                     color="#999999", va="bottom", ha="right")

    ax3.set_xlabel(xlabel)
    ax3.set_ylabel("P(CKD)")
    ax3.set_title("C)  CKD Probability vs Feature Value")
    ax3.set_ylim(-0.05, 1.05)
    ax3.yaxis.set_major_locator(MaxNLocator(6))

    # ── Overall title ──────────────────────────────────────────────────
    fig.suptitle(f"Figure {fig_num}.  {label}", fontsize=14, fontweight="bold",
                 y=1.02, color=TEXT_COLOR)

    fname = f"fig_{fig_num:02d}_{col}.png"
    fig.savefig(OUT_DIR / fname)
    plt.close()
    return fname


def make_correlation_heatmap(df_valid, y, features_list):
    """Create a correlation heatmap of all base features."""
    cols = [c for c in features_list if c in df_valid.columns]
    data = df_valid[cols].apply(pd.to_numeric, errors="coerce")
    data["CKD"] = y

    corr = data.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    labels = [c.replace("_", " ").title() for c in corr.columns]
    n = len(labels)

    # Plot lower triangle
    im = ax.imshow(np.where(mask, np.nan, corr.values), cmap="RdBu_r",
                   vmin=-1, vmax=1, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            if not mask[i, j]:
                val = corr.values[i, j]
                color = "white" if abs(val) > 0.6 else TEXT_COLOR
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6.5, color=color)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
    cbar.set_label("Pearson Correlation", fontsize=10)

    ax.set_title("Feature Correlation Matrix (with CKD Target)", fontsize=14,
                 fontweight="bold", pad=15)
    fig.savefig(OUT_DIR / "fig_correlation_heatmap.png")
    plt.close()
    print("  -> fig_correlation_heatmap.png")


def make_class_balance_figure(y):
    """Create a clean class balance figure."""
    n_ckd = (y == 1).sum()
    n_notckd = (y == 0).sum()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5),
                                    gridspec_kw={"width_ratios": [1, 1.2]})

    # Donut chart
    sizes = [n_ckd, n_notckd]
    colors = [CKD_BLUE, NOTCKD_CORAL]
    explode = (0.03, 0.03)
    wedges, texts, autotexts = ax1.pie(
        sizes, explode=explode, colors=colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.75,
        textprops=dict(fontsize=12, fontweight="bold", color="white"),
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2)
    )
    centre = plt.Circle((0, 0), 0.55, fc="white")
    ax1.add_artist(centre)
    ax1.text(0, 0, f"N={len(y)}", ha="center", va="center", fontsize=14,
             fontweight="bold", color=TEXT_COLOR)
    ax1.legend(["CKD", "non-CKD"], loc="lower center", fontsize=10,
               frameon=False, ncol=2, bbox_to_anchor=(0.5, -0.08))
    ax1.set_title("A)  Class Distribution", fontweight="bold")

    # Bar chart with counts
    bars = ax2.bar(["CKD", "non-CKD"], [n_ckd, n_notckd],
                   color=colors, alpha=0.8, edgecolor="white", linewidth=1.5, width=0.5)
    for bar, count in zip(bars, [n_ckd, n_notckd]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                 str(count), ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Number of Patients")
    ax2.set_title("B)  Class Counts", fontweight="bold")
    ax2.set_ylim(0, max(n_ckd, n_notckd) * 1.15)

    fig.suptitle("Dataset Class Balance", fontsize=14, fontweight="bold", y=1.02)
    fig.savefig(OUT_DIR / "fig_class_balance.png")
    plt.close()
    print("  -> fig_class_balance.png")


def make_missingness_heatmap(df_valid, y):
    """Create a per-feature missingness heatmap sorted by missingness."""
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

    ax.barh(y_pos + h, miss_df["CKD"], h, color=CKD_BLUE, alpha=0.8,
            label="CKD", edgecolor="white", linewidth=0.5)
    ax.barh(y_pos, miss_df["non-CKD"], h, color=NOTCKD_CORAL, alpha=0.8,
            label="non-CKD", edgecolor="white", linewidth=0.5)
    ax.barh(y_pos - h, miss_df["Overall"], h, color="#888888", alpha=0.5,
            label="Overall", edgecolor="white", linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(miss_df["Feature"], fontsize=9)
    ax.set_xlabel("Missing (%)", fontsize=11)
    ax.set_title("Missingness Rate by Feature and CKD Status", fontsize=14,
                 fontweight="bold")
    ax.legend(loc="lower right", frameon=True, fancybox=True, framealpha=0.9,
              edgecolor="#dddddd")

    # Add ratio annotations for top features
    for i, (_, row) in enumerate(miss_df.iterrows()):
        if row["CKD"] > 5 and row["non-CKD"] > 0:
            ratio = row["CKD"] / row["non-CKD"]
            ax.annotate(f"{ratio:.1f}x", xy=(row["CKD"] + 0.8, y_pos[i] + h),
                        fontsize=7, color="#666666", va="center")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig_missingness_detailed.png")
    plt.close()
    print("  -> fig_missingness_detailed.png")


def main():
    print("=" * 60)
    print("RenAI v1 — Polished EDA Visualizations")
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
