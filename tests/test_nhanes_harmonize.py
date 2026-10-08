import numpy as np
import pandas as pd
import pytest

from src.nhanes.constants import CREATININE_RECAL, WBC_SCALE
from src.nhanes.harmonize import harmonize, plausibility_report, recalibrate_creatinine


@pytest.fixture
def merged():
    return pd.DataFrame(
        {
            "SEQN": [1, 2, 3, 4],
            "cycle": ["D", "E", "P", "F"],
            "RIDAGEYR": [40, 50, 60, 17],
            "RIAGENDR": [1, 2, 1, 2],
            "RIDEXPRG": [np.nan, 2, np.nan, 3],
            "LBXSCR": [1.0, 1.0, 1.0, np.nan],
            "LBXWBCSI": [7.2, 5.0, 6.0, np.nan],
            "DIQ010": [1, 3, 9, 2],
            "BPXSY1": [120, 130, np.nan, np.nan],
            "BPXSY2": [124, np.nan, np.nan, np.nan],
            "BPXDI1": [80, 0, np.nan, np.nan],
            "BPXDI2": [82, 0, np.nan, np.nan],
            "BPXOSY1": [np.nan, np.nan, 110, np.nan],
            "BPXOSY2": [np.nan, np.nan, 112, np.nan],
            "BPXODI1": [np.nan, np.nan, 70, np.nan],
            "URDACT": [np.nan, 25.0, np.nan, np.nan],
            "URXUMA": [30.0, 10.0, 300.0, 5.0],
            "URXUCR": [100.0, 40.0, 100.0, 0.0],
            "WTMEC2YR": [10.0, 20.0, np.nan, 5.0],
            "WTMECPRP": [np.nan, np.nan, 30.0, np.nan],
        }
    )


def test_direct_renames_and_codes(merged):
    h = harmonize(merged)
    assert list(h["age"]) == [40, 50, 60, 17]
    assert list(h["sex"]) == ["male", "female", "male", "female"]
    assert list(h["sex_male"]) == [1.0, 0.0, 1.0, 0.0]
    # pregnancy: missing -> NaN, 2 -> 0, 3 (cannot ascertain) -> NaN
    assert np.isnan(h.loc[0, "pregnant"]) and h.loc[1, "pregnant"] == 0.0 and np.isnan(h.loc[3, "pregnant"])
    # diabetes: 1 -> 1, 3 borderline -> 0, 9 don't know -> NaN, 2 -> 0
    assert list(h["diabetes_mellitus"].fillna(-1)) == [1.0, 0.0, -1.0, 0.0]


def test_creatinine_recalibrated_only_for_2005_2006(merged):
    h = harmonize(merged)
    expected_d = CREATININE_RECAL["intercept"] + CREATININE_RECAL["slope"] * 1.0
    assert h.loc[0, "serum_creatinine"] == pytest.approx(expected_d)
    assert h.loc[0, "serum_creatinine_raw"] == 1.0
    assert h.loc[1, "serum_creatinine"] == 1.0
    assert h.loc[2, "serum_creatinine"] == 1.0
    assert np.isnan(h.loc[3, "serum_creatinine"])


def test_recalibrate_creatinine_preserves_nan_and_other_cycles():
    scr = pd.Series([0.8, np.nan, 0.8])
    cyc = pd.Series(["D", "D", "E"])
    out = recalibrate_creatinine(scr, cyc)
    assert out[0] == pytest.approx(-0.016 + 0.978 * 0.8)
    assert np.isnan(out[1])
    assert out[2] == 0.8
    assert scr[0] == 0.8  # input not mutated


def test_wbc_scaled_to_cells_per_ul(merged):
    h = harmonize(merged)
    assert h.loc[0, "white_blood_cell_count"] == pytest.approx(7.2 * WBC_SCALE)
    assert WBC_SCALE == 1000.0


def test_blood_pressure_mean_of_readings_zero_is_missing(merged):
    h = harmonize(merged)
    assert h.loc[0, "sbp"] == pytest.approx(122.0)
    assert h.loc[0, "blood_pressure"] == pytest.approx(81.0)
    assert h.loc[1, "sbp"] == pytest.approx(130.0)
    assert np.isnan(h.loc[1, "blood_pressure"])  # both diastolic readings were 0
    # P cycle uses the oscillometric columns
    assert h.loc[2, "sbp"] == pytest.approx(111.0)
    assert h.loc[2, "blood_pressure"] == pytest.approx(70.0)


def test_acr_uses_urdact_else_computed(merged):
    h = harmonize(merged)
    assert h.loc[0, "acr"] == pytest.approx(30.0 / 100.0 * 100)  # computed
    assert h.loc[0, "acr_source"] == "computed"
    assert h.loc[1, "acr"] == 25.0 and h.loc[1, "acr_source"] == "URDACT"
    assert h.loc[2, "acr"] == pytest.approx(300.0)
    assert np.isnan(h.loc[3, "acr"])  # urine creatinine 0 -> cannot compute
    assert h.loc[3, "acr_source"] is None or pd.isna(h.loc[3, "acr_source"])


def test_mec_weight_by_cycle(merged):
    h = harmonize(merged)
    assert list(h["mec_weight"]) == [10.0, 20.0, 30.0, 5.0]


def test_harmonize_does_not_mutate_input(merged):
    before = merged.copy(deep=True)
    harmonize(merged)
    pd.testing.assert_frame_equal(merged, before)


def test_plausibility_report_counts_hard_violations(merged):
    h = harmonize(merged).assign(sodium=[140.0, 99.0, 150.0, np.nan])  # 99 < hard min 100
    rep = plausibility_report(h)
    row = rep.set_index("column").loc["sodium"]
    assert row["hard_violations"] == 1 and row["n_checked"] == 3
