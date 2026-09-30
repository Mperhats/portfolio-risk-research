"""Tests for src.risk."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src import risk

SMALL = pd.Series([-0.10, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05])


def test_historical_var_on_known_series():
    # 10 observations; the 10% quantile with linear interpolation lies between
    # the 1st and 2nd order statistics: -0.10 + 0.9 * 0.05 = -0.055.
    assert risk.historical_var(SMALL, 0.90) == pytest.approx(0.055)
    # At 95% the quantile is -0.10 + 0.45 * 0.05 = -0.0775.
    assert risk.historical_var(SMALL, 0.95) == pytest.approx(0.0775)


def test_historical_es_is_mean_of_tail_and_exceeds_var():
    # 90% ES: observations <= 10% quantile (-0.055) -> only -0.10.
    assert risk.historical_es(SMALL, 0.90) == pytest.approx(0.10)
    # 80% ES: quantile at 20% is -0.10 + 1.8*0.05 = -0.01... check inclusive tail
    es80 = risk.historical_es(SMALL, 0.80)
    var80 = risk.historical_var(SMALL, 0.80)
    assert es80 >= var80
    for lvl in (0.9, 0.95, 0.99):
        assert risk.historical_es(SMALL, lvl) >= risk.historical_var(SMALL, lvl)


def test_var_monotone_in_confidence_level():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.standard_t(4, 5000) * 0.01)
    assert risk.historical_var(r, 0.99) > risk.historical_var(r, 0.95)
    assert risk.gaussian_var(r, 0.99) > risk.gaussian_var(r, 0.95)
    assert risk.student_t_var(r, 0.99) > risk.student_t_var(r, 0.95)


def test_gaussian_var_matches_closed_form():
    r = pd.Series([0.0, 0.02, -0.02, 0.01, -0.01])
    mu, sd = r.mean(), r.std(ddof=1)
    expected = -(mu + sd * stats.norm.ppf(0.05))
    assert risk.gaussian_var(r, 0.95) == pytest.approx(expected)
    assert risk.gaussian_es(r, 0.95) > risk.gaussian_var(r, 0.95)


def test_student_t_fit_recovers_heavy_tails():
    rng = np.random.default_rng(4)
    r = pd.Series(stats.t.rvs(4, loc=0.0005, scale=0.01, size=20000, random_state=rng))
    fit = risk.fit_student_t(r)
    assert 3.0 < fit.dof < 5.5
    assert fit.defensible
    # Student-t ES exceeds Student-t VaR and the closed form is finite.
    assert risk.student_t_es(r, 0.99, fit) > risk.student_t_var(r, 0.99, fit)


def test_invalid_level_rejected():
    with pytest.raises(ValueError):
        risk.historical_var(SMALL, 1.0)
    with pytest.raises(ValueError):
        risk.historical_es(SMALL, 0.2)


def test_bootstrap_shape_and_reproducibility():
    rng = np.random.default_rng(5)
    r = pd.Series(rng.normal(0, 0.01, 400))
    a = risk.bootstrap_var_es(r, 0.95, n_boot=50, block_size=5, seed=123)
    b = risk.bootstrap_var_es(r, 0.95, n_boot=50, block_size=5, seed=123)
    c = risk.bootstrap_var_es(r, 0.95, n_boot=50, block_size=5, seed=124)
    assert a.shape == (50, 2) and list(a.columns) == ["VaR", "ES"]
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)
    assert (a["ES"] >= a["VaR"]).all()


def test_bootstrap_iid_and_summary():
    rng = np.random.default_rng(6)
    r = pd.Series(rng.normal(0, 0.01, 400))
    s = risk.bootstrap_summary(r, levels=(0.95,), n_boot=100, block_size=1, seed=1)
    assert list(s.columns) == ["level", "measure", "estimate", "lower", "upper", "se"]
    assert len(s) == 2
    assert (s["lower"] <= s["estimate"]).all() and (s["estimate"] <= s["upper"]).all()


def test_risk_table_layout():
    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(0, 0.01, 1000))
    t = risk.risk_table(r, levels=(0.95, 0.99))
    assert len(t) == 12
    assert set(t["method"]) == {"Historical", "Gaussian", "Student-t"}
    assert (t["value"] > 0).all()


def test_stress_table_known_shock():
    idx = pd.bdate_range("2020-01-01", periods=50)
    rng = np.random.default_rng(8)
    ar = pd.DataFrame(rng.normal(0, 0.01, (50, 2)), index=idx, columns=["a", "b"])
    weights = {"a": 0.6, "b": 0.4}
    from src.config import StressScenario

    scen = (StressScenario("Half a", "a halves", {"a": -0.5}),)
    t = risk.stress_table(ar, weights, scen, value=1000.0)
    assert len(t) == 2  # scenario + worst historical day
    assert t.loc[0, "portfolio_return"] == pytest.approx(-0.3)
    assert t.loc[0, "loss_value"] == pytest.approx(300.0)
    worst = (ar["a"] * 0.6 + ar["b"] * 0.4).min()
    assert t.loc[1, "portfolio_return"] == pytest.approx(worst)
