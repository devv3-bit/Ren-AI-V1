# src/nhanes/constants.py
"""
Frozen configuration for Ren AI v2 (NHANES training + external validation).

Everything that defines the data sources, the harmonisation map, the cohort,
the label, the feature set and the temporal split is declared here, once,
before any model is fitted. Scripts, tests and the write-up import from this
module so the numbers reported are the numbers that were run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

SEED = 42

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_DIR = REPO_ROOT / "data" / "external" / "nhanes"
RAW_DIR = EXTERNAL_DIR / "raw"
MERGED_PATH = EXTERNAL_DIR / "merged.parquet"
HARMONIZED_PATH = EXTERNAL_DIR / "harmonized.parquet"
COHORT_PATH = EXTERNAL_DIR / "cohort.parquet"
MODELS_DIR = EXTERNAL_DIR / "models"  # frozen fitted models (git-ignored, re-fit deterministically)
REPORTS_DIR = REPO_ROOT / "reports" / "nhanes"
FIGURES_DIR = REPORTS_DIR / "figures"
UCI_PATH = REPO_ROOT / "data" / "raw" / "CKD.csv"

# ----------------------------------------------------------------------------
# NHANES cycles and files
# ----------------------------------------------------------------------------
# Cycle letter -> URL year, human label, MEC exam weight column.
CYCLES: Dict[str, Dict[str, object]] = {
    "D": {"year": 2005, "label": "2005-2006", "weight_col": "WTMEC2YR"},
    "E": {"year": 2007, "label": "2007-2008", "weight_col": "WTMEC2YR"},
    "F": {"year": 2009, "label": "2009-2010", "weight_col": "WTMEC2YR"},
    "G": {"year": 2011, "label": "2011-2012", "weight_col": "WTMEC2YR"},
    "H": {"year": 2013, "label": "2013-2014", "weight_col": "WTMEC2YR"},
    "I": {"year": 2015, "label": "2015-2016", "weight_col": "WTMEC2YR"},
    "P": {"year": 2017, "label": "2017-Mar 2020 (pre-pandemic)", "weight_col": "WTMECPRP"},
}
CYCLE_ORDER: List[str] = ["D", "E", "F", "G", "H", "I", "P"]
TRAIN_CYCLES: List[str] = ["D", "E", "F", "G", "H", "I"]  # 2005-2016
TEST_CYCLES: List[str] = ["P"]  # 2017-Mar 2020, touched once at the end

COMPONENTS: List[str] = ["DEMO", "BIOPRO", "ALB_CR", "CBC", "BPX", "DIQ", "BMX", "GHB"]
N_EXPECTED_FILES = len(CYCLE_ORDER) * len(COMPONENTS)  # 56
BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public"


def xpt_basename(cycle: str, component: str) -> str:
    """CDC file name (without extension) for a (cycle, component) pair."""
    if cycle == "P":
        comp = "BPXO" if component == "BPX" else component
        return f"P_{comp}"
    return f"{component}_{cycle}"


def xpt_path(cycle: str, component: str) -> Path:
    return RAW_DIR / f"{xpt_basename(cycle, component)}.xpt"


def xpt_url(cycle: str, component: str) -> str:
    year = CYCLES[cycle]["year"]
    return f"{BASE_URL}/{year}/DataFiles/{xpt_basename(cycle, component)}.xpt"


# ----------------------------------------------------------------------------
# Harmonisation: NHANES variable -> canonical internal name
# (canonical names match src/config.py where the variable exists there)
# ----------------------------------------------------------------------------
DIRECT_MAP: Dict[str, str] = {
    "RIDAGEYR": "age",                      # years
    "LBXSCR": "serum_creatinine",           # mg/dL (cycle D recalibrated, see CREATININE_RECAL)
    "LBXSBU": "bun",                        # mg/dL blood urea nitrogen
    "LBXSGL": "blood_glucose_random",       # mg/dL non-fasting serum glucose
    "LBXSNASI": "sodium",                   # mmol/L
    "LBXSKSI": "potassium",                 # mmol/L
    "LBXSCLSI": "chloride",                 # mmol/L
    "LBXSC3SI": "bicarbonate",              # mmol/L
    "LBXSAL": "serum_albumin",              # g/dL
    "LBXSUA": "uric_acid",                  # mg/dL
    "LBXSCA": "calcium",                    # mg/dL
    "LBXSPH": "phosphorus",                 # mg/dL
    "LBXHGB": "hemoglobin",                 # g/dL
    "LBXHCT": "packed_cell_volume",         # % (hematocrit)
    "LBXRBCSI": "red_blood_cell_count",     # million cells/uL
    "LBXWBCSI": "white_blood_cell_count",   # 1000 cells/uL in NHANES -> x1000 (WBC_SCALE)
    "LBXPLTSI": "platelets",                # 1000 cells/uL
    "LBXGH": "hba1c",                       # %
    "BMXBMI": "bmi",                        # kg/m2
    "URXUMA": "urine_albumin",              # ug/mL  (label only, never a feature)
    "URXUCR": "urine_creatinine",           # mg/dL  (label only, never a feature)
    "URDACT": "acr",                        # mg/g   (label only, never a feature)
    "SDMVPSU": "sdmvpsu",                   # survey design (secondary analyses only)
    "SDMVSTRA": "sdmvstra",
}

SEX_MAP: Dict[int, str] = {1: "male", 2: "female"}          # RIAGENDR
SEX_MALE_MAP: Dict[int, float] = {1: 1.0, 2: 0.0}
PREGNANCY_MAP: Dict[int, float] = {1: 1.0, 2: 0.0}          # RIDEXPRG; 3 = cannot ascertain -> NaN
DIABETES_MAP: Dict[int, float] = {1: 1.0, 2: 0.0, 3: 0.0}   # DIQ010; borderline -> no; 7/9 -> NaN

# CDC BIOPRO_D analytic note (confirmed against the documentation page):
# "Standard creatinine (mg/dL) = -0.016 + 0.978 X (NHANES 05-06 uncalibrated serum creatinine, mg/dL)"
CREATININE_RECAL: Dict[str, object] = {"cycle": "D", "intercept": -0.016, "slope": 0.978}
WBC_SCALE = 1000.0
ACR_FROM_COMPONENTS_FACTOR = 100.0  # (ug/mL) / (mg/dL) * 100 = mg/g

BP_READING_COLS: Dict[str, Dict[str, List[str]]] = {
    "default": {
        "sbp": ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"],
        "dbp": ["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"],
    },
    "P": {  # oscillometric protocol, P_BPXO
        "sbp": ["BPXOSY1", "BPXOSY2", "BPXOSY3"],
        "dbp": ["BPXODI1", "BPXODI2", "BPXODI3"],
    },
}

# ----------------------------------------------------------------------------
# Cohort and label (KDIGO, single visit)
# ----------------------------------------------------------------------------
ADULT_AGE = 18
EGFR_CKD_THRESHOLD = 60.0   # mL/min/1.73 m2 ; eGFR < 60  -> CKD (G3a+)
ACR_CKD_THRESHOLD = 30.0    # mg/g            ; ACR >= 30  -> CKD (A2+)
TARGET_COL = "ckd"
EARLY_SUBGROUP_COL = "early_subgroup"  # eGFR >= 60: CKD here can only come from albuminuria

# ----------------------------------------------------------------------------
# Features (declared before modelling; frozen)
# ----------------------------------------------------------------------------
BLOOD_TEST_FEATURES: List[str] = [
    "serum_creatinine", "bun", "blood_glucose_random", "sodium", "potassium",
    "chloride", "bicarbonate", "serum_albumin", "uric_acid", "calcium",
    "phosphorus", "hemoglobin", "packed_cell_volume", "red_blood_cell_count",
    "white_blood_cell_count", "platelets", "hba1c",
]
VITALS_DEMO_FEATURES: List[str] = [
    "age", "sex_male", "sbp", "blood_pressure", "bmi", "diabetes_mellitus",
]
ENGINEERED_FEATURES: List[str] = ["egfr_cr", "bun_creatinine_ratio"]
URINE_COLS: List[str] = ["urine_albumin", "urine_creatinine", "acr"]  # NEVER features

MODEL_FEATURES: List[str] = BLOOD_TEST_FEATURES + VITALS_DEMO_FEATURES + ENGINEERED_FEATURES

# Heavy-tailed -> log1p (same policy as v1 build_features.DEFAULT_LOG_COLS)
LOG_FEATURES: List[str] = ["serum_creatinine", "bun", "blood_glucose_random", "bun_creatinine_ratio"]

# Baselines
EGFR_ONLY_FEATURES: List[str] = ["egfr_cr"]
DEMOGRAPHIC_FEATURES: List[str] = ["age", "sex_male", "diabetes_mellitus", "sbp", "blood_pressure"]

# Harmonised feature set shared by NHANES and the UCI hospital data (Step 10).
# UCI has no sex column, no SBP, no BMI, no chloride/bicarbonate/albumin/uric
# acid/calcium/phosphorus/platelets/HbA1c. Hypertension is not mappable
# (NHANES BPQ questionnaire not in scope), so it is excluded.
HARMONIZED_FEATURES: List[str] = [
    "age", "blood_pressure", "blood_glucose_random", "bun", "serum_creatinine",
    "egfr_cr", "bun_creatinine_ratio", "sodium", "potassium", "hemoglobin",
    "packed_cell_volume", "white_blood_cell_count", "red_blood_cell_count",
    "diabetes_mellitus",
]
UREA_TO_BUN_DIVISOR = 2.14  # BUN (mg/dL) = urea (mg/dL) x 28/60

# ----------------------------------------------------------------------------
# Modelling grid (tuned by 5-fold stratified CV on TRAIN only)
# ----------------------------------------------------------------------------
N_SPLITS = 5
LOGISTIC_C_GRID: List[float] = [0.001, 0.01, 0.1, 1.0, 10.0]
HGB_GRID: Dict[str, List[object]] = {
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 6],
    "max_iter": [100, 300],
}
N_BOOTSTRAP = 1000
