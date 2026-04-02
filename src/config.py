#src/config.py
#src/config.py is used to setup the internal schema for the project
#raw dataset columns-> internal coloumn names to setup for preprocessing,setting up ranges for CKD parameters eg, creatinine etc.
# src/config.py

# Raw dataset column -> canonical internal column name
I_SCHEMA = {
    "age": "age",
    "bp": "blood_pressure",
    "sg": "specific_gravity",
    "al": "albumin",
    "su": "sugar",
    "rbc": "urine_red_blood_cells",
    "pc": "pus_cell",
    "pcc": "pus_cell_clumps",
    "ba": "bacteria",
    "bgr": "blood_glucose_random",
    "bu": "blood_urea",
    "sc": "serum_creatinine",
    "sod": "sodium",
    "pot": "potassium",
    "hemo": "hemoglobin",
    "pcv": "packed_cell_volume",
    "wc": "white_blood_cell_count",
    "wbcc": "white_blood_cell_count",
    "rc": "red_blood_cell_count",
    "rbcc": "red_blood_cell_count",
    "htn": "hypertension",
    "dm": "diabetes_mellitus",
    "cad": "coronary_artery_disease",
    "appet": "appetite",
    "pe": "pedal_edema",
    "ane": "anemia",
    "classification": "ckd_class",
    "class": "ckd_class",

}

# Which canonical columns should be numeric
NUMERIC_COLS = [
    "age", "blood_pressure", "specific_gravity", "albumin", "sugar",
    "blood_glucose_random", "blood_urea", "serum_creatinine",
    "sodium", "potassium", "hemoglobin", "packed_cell_volume",
    "white_blood_cell_count", "red_blood_cell_count",
]


BINARY_COLS = [
    "hypertension", "diabetes_mellitus", "coronary_artery_disease",
    "pedal_edema", "anemia"
]


VALIDATION_RULES = {
    "age": {"hard": (0, 120), "soft": (10, 95)},

    "blood_pressure": {"hard": (30, 300)},
    "serum_creatinine": {"hard": (0, 30)},
    "blood_urea": {"hard": (0, 300)},
    "sodium": {"hard": (100, 200)},
    "potassium": {"hard": (1, 10)},
    "hemoglobin": {"hard": (1, 25)},
    "blood_glucose_random": {"hard": (20, 1000)},
    "packed_cell_volume": {"hard": (0, 80)},
    "white_blood_cell_count": {"hard": (1000, 50000)},
    "red_blood_cell_count": {"hard": (0,10)},
}


OPTIONAL_ENHANCED_BIOMARKERS = {
    "hba1c": {"hard": (3, 20)},          # %
    "cystatin_c": {"hard": (0, 10)},     # mg/L (broad sanity)
    "crp": {"hard": (0, 500)},           # mg/L (broad sanity)
}
