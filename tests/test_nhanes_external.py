from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.derive import egfr_cr
from src.nhanes.constants import HARMONIZED_FEATURES, UCI_PATH, UREA_TO_BUN_DIVISOR
from src.nhanes.external import feature_availability, load_uci_harmonized, uci_target, yes_no_to_float

needs_uci = pytest.mark.skipif(not Path(UCI_PATH).exists(), reason="UCI CKD.csv not present")


def test_yes_no_mapping_handles_whitespace_and_unknowns():
    s = pd.Series(["yes", " no", "\tyes", "NO", "?", None])
    out = yes_no_to_float(s)
    assert list(out.fillna(-1)) == [1.0, 0.0, 1.0, 0.0, -1.0, -1.0]


@needs_uci
def test_uci_harmonized_has_all_shared_features_and_label():
    df = load_uci_harmonized("male")
    assert len(df) == 399  # 400 lines incl. header -> 399 usable patients
    missing = [c for c in HARMONIZED_FEATURES if c not in df.columns]
    assert not missing
    y = uci_target(df)
    assert len(y) == 399 and set(np.unique(y)) == {0, 1}
    # urea -> BUN
    row = df.dropna(subset=["blood_urea"]).iloc[0]
    assert row["bun"] == pytest.approx(row["blood_urea"] / UREA_TO_BUN_DIVISOR)
    assert UREA_TO_BUN_DIVISOR == 2.14
    avail = feature_availability(df, HARMONIZED_FEATURES)
    # UCI is sparse for some labs (RBC count is missing for ~1/3 of rows) but every
    # shared feature is observed for most patients; the NHANES imputer handles the rest.
    assert (avail["n_nonmissing"] >= 260).all()
    assert avail.set_index("feature").loc["serum_creatinine", "n_nonmissing"] >= 380


@needs_uci
def test_uci_sex_assumption_changes_egfr_only():
    male = load_uci_harmonized("male")
    female = load_uci_harmonized("female")
    i = male.dropna(subset=["serum_creatinine", "age"]).index[0]
    assert male.loc[i, "egfr_cr"] == pytest.approx(egfr_cr(male.loc[i, "serum_creatinine"], male.loc[i, "age"], "male"))
    assert female.loc[i, "egfr_cr"] == pytest.approx(egfr_cr(male.loc[i, "serum_creatinine"], male.loc[i, "age"], "female"))
    assert female.loc[i, "egfr_cr"] < male.loc[i, "egfr_cr"]
    pd.testing.assert_series_equal(male["bun"], female["bun"])
    with pytest.raises(ValueError):
        load_uci_harmonized("unknown")
