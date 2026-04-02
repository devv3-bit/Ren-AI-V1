# ==========================
# EDA-2 (CKD vs non-CKD)
# Copy-paste into day5_eda.ipynb
# ==========================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# Use your ONE pipeline entry point (good system hygiene)
from src.pipeline.run_pipeline import run_pipeline

# --------------------------
# 0) Load data through pipeline
# --------------------------
DATA_PATH = Path("data/raw/CKD.csv")
assert DATA_PATH.exists(), f"Missing file: {DATA_PATH.resolve()}"

out = run_pipeline(str(DATA_PATH), require_target=True)
df = out.df_valid.copy()

print("Loaded df_valid shape:", df.shape)
print("Target col:", "ckd_class")
print(df["ckd_class"].value_counts(dropna=False))


# --------------------------
# 1) Helpers
# --------------------------
BASE_FEATURES = [
    "age",
    "blood_pressure",
    "specific_gravity",
    "albumin",
    "sugar",
    "blood_glucose_random",
    "blood_urea",
    "serum_creatinine",
    "sodium",
    "potassium",
    "hemoglobin",
    "packed_cell_volume",
    "white_blood_cell_count",
    "red_blood_cell_count",
]

DERIVED_FEATURES = [
    "egfr_cr",
    "urea_creatinine_ratio",
    "egfr_cys",
    "egfr_cr_cys",
    "egfr_discordance",
]

ENHANCED_FEATURES = [
    "hba1c",
    "cystatin_c",
    "crp",
]

ALL_FEATURES = BASE_FEATURES + DERIVED_FEATURES + ENHANCED_FEATURES


def normalize_target(s: pd.Series) -> pd.Series:
    """Normalize target to {ckd, notckd} strings."""
    x = s.astype(str).str.strip().str.lower()
    # common variants
    x = x.replace({
        "1": "ckd", "true": "ckd", "yes": "ckd", "ckd": "ckd",
        "0": "notckd", "false": "notckd", "no": "notckd", "notckd": "notckd"
    })
    return x

def split_by_class(df: pd.DataFrame, target: str = "ckd_class"):
    y = normalize_target(df[target])
    ckd_df = df[y == "ckd"]
    non_ckd_df = df[y == "notckd"]
    return ckd_df, non_ckd_df

def missing_pct(s: pd.Series) -> float:
    return float(s.isna().mean() * 100)

def safe_divide(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return a / b.replace(0, np.nan)

def compute_missingness(df: pd.DataFrame, feature: str, target: str = "ckd_class"):
    if feature not in df.columns:
        return 100.0, 100.0, 100.0

    s_all = pd.to_numeric(df[feature], errors="coerce")
    if s_all.notna().sum() == 0:
        return 100.0, 100.0, 100.0

    y = normalize_target(df[target])
    s_ckd = pd.to_numeric(df.loc[y == "ckd", feature], errors="coerce")
    s_non = pd.to_numeric(df.loc[y == "notckd", feature], errors="coerce")

    miss_all = missing_pct(s_all)
    miss_ckd = missing_pct(s_ckd) if len(s_ckd) else 100.0
    miss_non = missing_pct(s_non) if len(s_non) else 100.0
    return miss_all, miss_ckd, miss_non


def compute_medians(df: pd.DataFrame, feature: str, target: str = "ckd_class"):
    if feature not in df.columns:
        return "NA", "NA", "NA"

    s_all = pd.to_numeric(df[feature], errors="coerce")
    if s_all.notna().sum() == 0:
        return "NA", "NA", "NA"

    y = normalize_target(df[target])
    s_ckd = pd.to_numeric(df.loc[y == "ckd", feature], errors="coerce")
    s_non = pd.to_numeric(df.loc[y == "notckd", feature], errors="coerce")

    med_ckd = np.nanmedian(s_ckd) if s_ckd.notna().any() else np.nan
    med_non = np.nanmedian(s_non) if s_non.notna().any() else np.nan

    med_ckd_out = f"{med_ckd:.4g}" if np.isfinite(med_ckd) else "NA"
    med_non_out = f"{med_non:.4g}" if np.isfinite(med_non) else "NA"

    if np.isfinite(med_ckd) and np.isfinite(med_non):
        if med_ckd > med_non:
            direction = "CKD > non-CKD"
        elif med_ckd < med_non:
            direction = "CKD < non-CKD"
        else:
            direction = "equal"
    else:
        direction = "NA"

    return med_ckd_out, med_non_out, direction

def binned_ckd_rate(df: pd.DataFrame, feature: str, target="ckd_class", bins=10):
    """For monotonicity: bin feature into quantiles and compute CKD rate per bin."""
    tmp = df[[feature, target]].copy()
    tmp[target] = normalize_target(tmp[target])
    tmp = tmp.dropna(subset=[feature, target])
    if tmp.empty:
        return None

    # If too few unique values, quantile binning may fail
    try:
        tmp["bin"] = pd.qcut(tmp[feature], q=bins, duplicates="drop")
    except Exception:
        return None

    rate = (tmp[target] == "ckd").groupby(tmp["bin"]).mean()
    centers = tmp.groupby("bin")[feature].median()
    return pd.DataFrame({"bin_center": centers.values, "ckd_rate": rate.values}).sort_values("bin_center")


# --------------------------
# 2) The EDA function (answers B1–B4 visually)
# --------------------------
def eda_feature(df: pd.DataFrame, feature: str, target: str = "ckd_class", bins: int = 10):
    ckd_df, non_ckd_df = split_by_class(df, target)

    print("\n" + "="*60)
    print(f"FEATURE: {feature}")
    print("="*60)

    # Basic availability
    if feature not in df.columns:
        print(f"❌ Feature '{feature}' not in df columns.")
        return

    # Ensure numeric (for numeric features)
    s_all = pd.to_numeric(df[feature], errors="coerce")
    s_ckd = pd.to_numeric(ckd_df[feature], errors="coerce")
    s_non = pd.to_numeric(non_ckd_df[feature], errors="coerce")

    n_ckd = int(s_ckd.notna().sum())
    n_non = int(s_non.notna().sum())
    n_total = n_ckd + n_non

    # B1 Direction (median)
    med_ckd = float(np.nanmedian(s_ckd)) if n_ckd > 0 else np.nan
    med_non = float(np.nanmedian(s_non)) if n_non > 0 else np.nan
    print("B1 Direction (median):")
    if n_ckd > 0:
        print(f"  median(CKD)     = {med_ckd:.4g}")
    else:
        print("  median(CKD)     = NA")
    if n_non > 0:
        print(f"  median(non-CKD) = {med_non:.4g}")
    else:
        print("  median(non-CKD) = NA")
    if np.isfinite(med_ckd) and np.isfinite(med_non):
        if med_ckd > med_non:
            print("  👉 Typical value is HIGHER in CKD.")
        elif med_ckd < med_non:
            print("  👉 Typical value is HIGHER in non-CKD.")
        else:
            print("  👉 Medians are equal (rare).")

    # Missingness (useful for EDA-3 but good to see now)
    print("Missingness:")
    print(f"  missing% overall  = {missing_pct(df[feature]):.2f}%")
    print(f"  missing% CKD      = {missing_pct(ckd_df[feature]):.2f}%")
    print(f"  missing% non-CKD  = {missing_pct(non_ckd_df[feature]):.2f}%")

    if n_total == 0:
        print(f"SKIP PLOTS: {feature} is 100% missing")
        return

    # Plots: distribution overlay + boxplot
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    skewed_feature = feature in {"blood_pressure", "specific_gravity", "albumin", }
    eps = 1e-3
    s_ckd_plot = s_ckd + eps if skewed_feature else s_ckd
    s_non_plot = s_non + eps if skewed_feature else s_non

    # B2 Separation: histogram overlays
    plotted = False
    if n_ckd >= 2:
        sns.histplot(s_ckd_plot.dropna(), kde=True, stat="density", ax=axes[0], label="CKD", alpha=0.55)
        plotted = True
    if n_non >= 2:
        sns.histplot(s_non_plot.dropna(), kde=True, stat="density", ax=axes[0], label="non-CKD", alpha=0.55)
        plotted = True
    if plotted:
        axes[0].set_title(f"B2 Separation (distribution): {feature}")
        # Fixed y-scale for visual clarity only; not a probability statement.
        axes[0].set_ylim(0, 100)
        if skewed_feature:
            axes[0].set_xscale("log")
        else:
            axes[0].set_xlim(0, 100)
        if feature == "egfr_cr":
            axes[0].set_xlim(eps, 120)
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, "Not enough data for distribution",
                     ha="center", va="center")
        axes[0].set_axis_off()

    # B3 Stability/outliers: boxplot
    tmp = df[[feature, target]].copy()
    tmp[target] = normalize_target(tmp[target])
    tmp[feature] = pd.to_numeric(tmp[feature], errors="coerce")
    n_ckd_box = int(tmp.loc[tmp[target] == "ckd", feature].notna().sum())
    n_non_box = int(tmp.loc[tmp[target] == "notckd", feature].notna().sum())
    if n_ckd_box >= 2 and n_non_box >= 2:
        tmp_box = tmp
    elif n_ckd_box >= 2:
        tmp_box = tmp[tmp[target] == "ckd"]
    elif n_non_box >= 2:
        tmp_box = tmp[tmp[target] == "notckd"]
    else:
        tmp_box = None

    if tmp_box is None:
        axes[1].text(0.5, 0.5, "Not enough data for boxplot",
                     ha="center", va="center")
        axes[1].set_axis_off()
    else:
        sns.boxplot(data=tmp_box, x=target, y=feature, ax=axes[1])
        axes[1].set_title(f"B3 Outliers / stability: {feature}")
        if skewed_feature:
            y_max = np.nanpercentile(tmp_box[feature], 99)
            if np.isfinite(y_max) and y_max > 0:
                axes[1].set_ylim(0, y_max)
        else:
            axes[1].set_ylim(0, 100)
        axes[1].set_xlim(0, 4)
        axes[1].set_xticks(np.arange(0, 5, 1))
        axes[1].set_xlabel(target)
        axes[1].set_ylabel(feature)

    # B4 Monotonicity: binned CKD rate vs feature
    if n_total < 10:
        print("Warning: low data for monotonicity")
    br = binned_ckd_rate(df, feature, target=target, bins=bins)
    if br is None:
        axes[2].text(0.5, 0.5, "Not enough data to bin\n(or too few unique values)",
                     ha="center", va="center")
        axes[2].set_axis_off()
    else:
        axes[2].plot(br["bin_center"], br["ckd_rate"], marker="o")
        axes[2].set_ylim(-0.05, 1.05)
        axes[2].set_title(f"B4 Monotonicity: CKD rate vs {feature}")
        axes[2].set_xlabel(feature)
        axes[2].set_ylabel("P(CKD)")

    plt.tight_layout()
    plt.show()

    print("B5 Verdict (YOU decide): Strong / Moderate / Weak")
    print("Tip: if histograms overlap a lot → not strong. If boxplot shift is big → stronger.")

def _format_pct(val: float) -> str:
    return f"{val:.2f}%"


def write_missingness_report(df: pd.DataFrame, features, target: str = "ckd_class"):
    report_path = Path("texts/missingness_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for feat in features:
        miss_all, miss_ckd, miss_non = compute_missingness(df, feat, target=target)
        lines.append(f"Feature: {feat}")
        lines.append(f"- Missing % (overall): {_format_pct(miss_all)}")
        lines.append(f"- Missing % (CKD): {_format_pct(miss_ckd)}")
        lines.append(f"- Missing % (non-CKD): {_format_pct(miss_non)}")
        lines.append("")

    report_path.write_text("\n".join(lines))


def write_median_report(df: pd.DataFrame, features, target: str = "ckd_class"):
    report_path = Path("texts/median_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for feat in features:
        med_ckd, med_non, direction = compute_medians(df, feat, target=target)
        lines.append(f"Feature: {feat}")
        lines.append(f"- Median (CKD): {med_ckd}")
        lines.append(f"- Median (non-CKD): {med_non}")
        lines.append(f"- Direction: {direction}")
        lines.append("")

    report_path.write_text("\n".join(lines))


# --------------------------
# 3) Standalone EDA block for egfr_cr
# --------------------------
def eda_egfr_cr(df: pd.DataFrame, feature: str = "egfr_cr", target: str = "ckd_class", bins: int = 10):
    ckd_df, non_ckd_df = split_by_class(df, target)
    print("\n" + "="*60)
    print(f"FEATURE: {feature} (standalone)")
    print("="*60)

    if feature not in df.columns:
        print(f"❌ Feature '{feature}' not in df columns.")
        return

    s_all = pd.to_numeric(df[feature], errors="coerce")
    s_ckd = pd.to_numeric(ckd_df[feature], errors="coerce")
    s_non = pd.to_numeric(non_ckd_df[feature], errors="coerce")

    n_ckd = int(s_ckd.notna().sum())
    n_non = int(s_non.notna().sum())
    n_total = int(s_all.notna().sum())

    print("Missingness:")
    print(f"  missing% overall  = {missing_pct(s_all):.2f}%")
    print(f"  missing% CKD      = {missing_pct(s_ckd):.2f}%")
    print(f"  missing% non-CKD  = {missing_pct(s_non):.2f}%")

    med_ckd = float(np.nanmedian(s_ckd)) if n_ckd > 0 else np.nan
    med_non = float(np.nanmedian(s_non)) if n_non > 0 else np.nan
    print("Medians (ignoring NaNs):")
    print(f"  median(CKD)     = {med_ckd:.4g}" if np.isfinite(med_ckd) else "  median(CKD)     = NA")
    print(f"  median(non-CKD) = {med_non:.4g}" if np.isfinite(med_non) else "  median(non-CKD) = NA")

    if n_total == 0:
        print("SKIP PLOTS: egfr_cr is 100% missing")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    # A) distribution overlay CKD vs non-CKD
    plotted = False
    if n_ckd >= 2:
        sns.histplot(s_ckd.dropna(), kde=True, stat="density", ax=axes[0], label="CKD", alpha=0.55)
        plotted = True
    if n_non >= 2:
        sns.histplot(s_non.dropna(), kde=True, stat="density", ax=axes[0], label="non-CKD", alpha=0.55)
        plotted = True
    if plotted:
        axes[0].set_title("A) egfr_cr distribution (CKD vs non-CKD)")
        p1, p99 = np.nanpercentile(s_all, [1, 99])
        if np.isfinite(p1) and np.isfinite(p99) and p99 > p1:
            axes[0].set_xlim(p1, p99)
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, "Not enough data for distribution",
                     ha="center", va="center")
        axes[0].set_axis_off()

    # B) boxplot CKD vs non-CKD only if both groups have >=5 non-NaN values
    if n_ckd >= 5 and n_non >= 5:
        tmp = df[[feature, target]].copy()
        tmp[target] = normalize_target(tmp[target])
        tmp[feature] = pd.to_numeric(tmp[feature], errors="coerce")
        sns.boxplot(data=tmp, x=target, y=feature, ax=axes[1])
        axes[1].set_title("B) egfr_cr boxplot (CKD vs non-CKD)")
    else:
        axes[1].text(0.5, 0.5, "Boxplot skipped: <5 non-NaN per group",
                     ha="center", va="center")
        axes[1].set_axis_off()

    # C) monotonicity curve via ~10 quantile bins
    br = binned_ckd_rate(df, feature, target=target, bins=bins)
    if br is None or br.empty:
        axes[2].text(0.5, 0.5, "Not enough data to bin\n(or too few unique values)",
                     ha="center", va="center")
        axes[2].set_axis_off()
    else:
        axes[2].plot(br["bin_center"], br["ckd_rate"], marker="o")
        axes[2].set_ylim(-0.05, 1.05)
        axes[2].set_title("C) egfr_cr monotonicity (P(CKD) vs bin center)")
        axes[2].set_xlabel(feature)
        axes[2].set_ylabel("P(CKD)")

    plt.tight_layout()
    plt.show()


# --------------------------
# 4) Run EDA-2 for your priority NUMERIC features
# --------------------------
numeric_features = [
    "blood_urea",
    "sodium",
    "potassium",
    "packed_cell_volume",
    "white_blood_cell_count",
    "red_blood_cell_count",
    "urea_creatinine_ratio",
]

for feat in numeric_features:
    eda_feature(df, feat, bins=10)

eda_egfr_cr(df, bins=10)

# --------------------------
# 5) Export missingness + median summaries
# --------------------------
write_missingness_report(df, ALL_FEATURES, target="ckd_class")
write_median_report(df, ALL_FEATURES, target="ckd_class")
