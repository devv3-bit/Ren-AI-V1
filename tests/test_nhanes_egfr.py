import numpy as np
import pandas as pd
import pytest

from src.features.derive import egfr_cr
from src.nhanes.egfr import egfr_cr_vectorized


def test_vectorized_matches_scalar_on_random_inputs():
    rng = np.random.default_rng(42)
    n = 5000
    scr = rng.uniform(0.3, 15.0, n)
    age = rng.uniform(18, 90, n)
    sex = rng.choice(["male", "female", "M", "f", " Female "], n)

    vec = egfr_cr_vectorized(scr, age, sex)
    ref = np.array([egfr_cr(s, a, x) for s, a, x in zip(scr, age, sex)])

    np.testing.assert_allclose(vec, ref, rtol=1e-12, atol=0.0)


def test_vectorized_handles_both_branches_of_the_min_max_equation():
    # creatinine below and above the sex-specific kappa (0.7 female, 0.9 male)
    scr = np.array([0.5, 0.7, 0.9, 1.3, 4.0, 0.5, 0.7, 0.9, 1.3, 4.0])
    age = np.full(10, 55.0)
    sex = ["female"] * 5 + ["male"] * 5
    vec = egfr_cr_vectorized(scr, age, sex)
    ref = [egfr_cr(s, a, x) for s, a, x in zip(scr, age, sex)]
    np.testing.assert_allclose(vec, ref, rtol=1e-12)


def test_vectorized_accepts_series_and_returns_nan_for_missing_or_invalid():
    df = pd.DataFrame(
        {
            "scr": [1.0, np.nan, 1.0, 0.0, 1.0, -1.0],
            "age": [50, 50, np.nan, 50, 50, 50],
            "sex": ["male", "female", "male", "male", "other", "female"],
        }
    )
    out = egfr_cr_vectorized(df["scr"], df["age"], df["sex"])
    assert out[0] == pytest.approx(egfr_cr(1.0, 50, "male"))
    assert np.isnan(out[1:]).all()


def test_vectorized_rejects_length_mismatch():
    with pytest.raises(ValueError):
        egfr_cr_vectorized([1.0, 1.0], [50.0], ["male", "male"])


def test_sex_changes_the_estimate_as_expected():
    male = egfr_cr_vectorized([1.0], [40.0], ["male"])[0]
    female = egfr_cr_vectorized([1.0], [40.0], ["female"])[0]
    assert female < male  # same creatinine -> lower eGFR for women (kappa 0.7)
    assert 90 < male < 105
