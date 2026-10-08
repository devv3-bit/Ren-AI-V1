import numpy as np
import pandas as pd
import pytest

from src.features.derive import egfr_cr
from src.nhanes.cohort import apply_cohort_filters, split_label
from src.nhanes.label import add_egfr, add_labels, prevalence_summary


def _frame():
    # creatinine chosen so eGFR is clearly above/below 60 for a 50-year-old male
    return pd.DataFrame(
        {
            "cycle": ["D", "E", "I", "P", "P", "F", "G", "H"],
            "age": [50, 50, 50, 50, 50, 17, 30, 50],
            "sex": ["male"] * 8,
            "pregnant": [np.nan, 0.0, np.nan, np.nan, np.nan, np.nan, 1.0, np.nan],
            "serum_creatinine": [0.9, 2.5, 0.9, 2.5, 0.9, 0.9, 0.9, np.nan],
            "acr": [10.0, 10.0, 45.0, 300.0, 29.9, 10.0, 10.0, 10.0],
        }
    )


def test_cohort_flow_counts_each_exclusion():
    cohort, steps = apply_cohort_filters(_frame())
    assert [s.n_remaining for s in steps] == [8, 7, 6, 5]
    assert [s.n_excluded for s in steps] == [0, 1, 1, 1]
    assert len(cohort) == 5
    assert 17 not in cohort["age"].values  # minor removed
    assert not cohort["pregnant"].eq(1.0).any()  # pregnant removed, unknown kept
    assert cohort["serum_creatinine"].notna().all() and cohort["acr"].notna().all()


def test_cohort_filters_do_not_mutate_input():
    df = _frame()
    before = df.copy(deep=True)
    apply_cohort_filters(df)
    pd.testing.assert_frame_equal(df, before)


def test_kdigo_label_and_early_subgroup():
    cohort, _ = apply_cohort_filters(_frame())
    df = add_labels(add_egfr(cohort))
    assert df.loc[0, "egfr_cr"] == pytest.approx(egfr_cr(0.9, 50, "male"))
    # row0: eGFR high, ACR 10  -> no CKD, early subgroup
    # row1: eGFR low,  ACR 10  -> CKD via eGFR, not early subgroup
    # row2: eGFR high, ACR 45  -> CKD via albuminuria only, early subgroup
    # row3: eGFR low,  ACR 300 -> CKD via both
    # row4: eGFR high, ACR 29.9 -> no CKD (threshold is >= 30)
    assert list(df["ckd"]) == [0, 1, 1, 1, 0]
    assert list(df["early_subgroup"]) == [True, False, True, False, True]
    assert list(df["egfr_low"]) == [False, True, False, True, False]
    assert list(df["albuminuria"]) == [False, False, True, True, False]
    assert df["ckd"].dtype.kind == "i"


def test_label_thresholds_are_inclusive_exclusive_as_kdigo():
    df = pd.DataFrame({"egfr_cr": [60.0, 59.999, 60.0], "acr": [30.0, 29.999, 29.999]})
    out = add_labels(df)
    assert list(out["ckd"]) == [1, 1, 0]
    assert list(out["early_subgroup"]) == [True, False, True]


def test_label_refuses_missing_components():
    with pytest.raises(ValueError):
        add_labels(pd.DataFrame({"egfr_cr": [np.nan], "acr": [10.0]}))


def test_prevalence_summary():
    cohort, _ = apply_cohort_filters(_frame())
    df = add_labels(add_egfr(cohort))
    s = prevalence_summary(df)
    assert s["n"] == 5 and s["n_ckd"] == 3
    assert s["prevalence"] == pytest.approx(3 / 5)
    assert s["n_early_subgroup"] == 3 and s["n_ckd_early_subgroup"] == 1
    assert s["prevalence_early_subgroup"] == pytest.approx(1 / 3)
    assert s["n_both"] == 1


def test_temporal_split_label():
    out = split_label(pd.Series(["D", "I", "P"]))
    assert list(out) == ["train", "train", "test"]
