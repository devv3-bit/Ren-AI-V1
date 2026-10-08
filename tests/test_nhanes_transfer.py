import numpy as np
import pandas as pd
import pytest

from src.nhanes.models import build_estimator
from src.nhanes.transfer import RangeClipper, make_transfer_logistic


def test_range_clipper_clips_to_train_quantiles_and_keeps_nan():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 2))
    clip = RangeClipper(lower=0.01, upper=0.99).fit(X)
    out = clip.transform(np.array([[100.0, -100.0], [np.nan, 0.0]]))
    assert out[0, 0] == pytest.approx(np.quantile(X[:, 0], 0.99))
    assert out[0, 1] == pytest.approx(np.quantile(X[:, 1], 0.01))
    assert np.isnan(out[1, 0]) and out[1, 1] == 0.0
    with pytest.raises(ValueError):
        clip.transform(np.zeros((1, 3)))
    with pytest.raises(ValueError):
        RangeClipper(lower=0.9, upper=0.1).fit(X)


def test_transfer_pipeline_has_no_missing_indicators_and_handles_out_of_range():
    rng = np.random.default_rng(1)
    n = 500
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = (X["a"] + 0.5 * rng.normal(size=n) > 0).astype(int).to_numpy()
    est = make_transfer_logistic(1.0).fit(X, y)
    assert est.named_steps["clf"].coef_.shape == (1, 2)  # no indicator columns
    # an extreme, partly-missing row still gets a finite probability driven by the clipped value
    p = est.predict_proba(pd.DataFrame({"a": [1e6, -1e6, np.nan], "b": [np.nan, 0.0, 0.0]}))[:, 1]
    assert np.isfinite(p).all() and p[0] > 0.9 and p[1] < 0.1
    noclip = make_transfer_logistic(1.0, clip=False)
    assert "clip" not in noclip.named_steps
    assert "clip" in build_estimator("transfer_logistic", {"C": 0.1}).named_steps
    assert "clip" not in build_estimator("transfer_logistic_noclip", {"C": 0.1}).named_steps
