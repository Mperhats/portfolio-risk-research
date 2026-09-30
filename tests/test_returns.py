"""Tests for src.returns."""

import numpy as np
import pandas as pd
import pytest

from src import returns as R


@pytest.fixture
def prices() -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=5)
    return pd.DataFrame({"x": [100.0, 110.0, 99.0, 99.0, 108.9], "y": [50.0, 50.0, 55.0, 44.0, 44.0]}, index=idx)


def test_simple_returns_known_values(prices):
    r = R.simple_returns(prices)
    assert len(r) == 4
    np.testing.assert_allclose(r["x"].to_numpy(), [0.10, -0.10, 0.0, 0.10])
    np.testing.assert_allclose(r["y"].to_numpy(), [0.0, 0.10, -0.20, 0.0])


def test_log_returns_match_log_of_gross(prices):
    lr = R.log_returns(prices)
    r = R.simple_returns(prices)
    np.testing.assert_allclose(lr.to_numpy(), np.log1p(r.to_numpy()))


def test_portfolio_returns_weighted_sum(prices):
    r = R.simple_returns(prices)
    p = R.portfolio_returns(r, {"x": 0.75, "y": 0.25})
    expected = 0.75 * r["x"] + 0.25 * r["y"]
    np.testing.assert_allclose(p.to_numpy(), expected.to_numpy())
    assert p.name == "portfolio"


def test_portfolio_returns_rejects_bad_weights(prices):
    r = R.simple_returns(prices)
    with pytest.raises(ValueError):
        R.portfolio_returns(r, {"x": 0.7, "y": 0.2})
    with pytest.raises(KeyError):
        R.portfolio_returns(r, {"x": 0.5, "z": 0.5})


def test_cumulative_growth_and_return():
    r = pd.Series([0.10, -0.10, 0.05])
    g = R.cumulative_growth(r)
    np.testing.assert_allclose(g.to_numpy(), [1.10, 0.99, 1.0395])
    assert R.cumulative_return(r) == pytest.approx(0.0395)


def test_annualisation():
    r = pd.Series(np.full(252, 0.001))
    assert R.annualized_mean(r) == pytest.approx(0.252)
    assert R.annualized_volatility(r) == pytest.approx(0.0)
    r2 = pd.Series([0.01, -0.01] * 126)
    assert R.annualized_volatility(r2) == pytest.approx(r2.std(ddof=1) * np.sqrt(252))


def test_max_drawdown_simple_path():
    # 100 -> 120 -> 60 -> 90 -> 130: max drawdown is 60/120 - 1 = -50%
    idx = pd.bdate_range("2021-01-04", periods=5)
    px = pd.Series([100.0, 120.0, 60.0, 90.0, 130.0], index=idx)
    r = R.simple_returns(px)
    s = R.max_drawdown(r)
    assert s.max_drawdown == pytest.approx(-0.5)
    assert s.peak_date == idx[1]
    assert s.trough_date == idx[2]
    assert s.recovery_date == idx[4]
    assert s.duration_days == 1


def test_max_drawdown_monotonic_path_is_zero():
    idx = pd.bdate_range("2021-01-04", periods=4)
    r = R.simple_returns(pd.Series([1.0, 1.1, 1.2, 1.3], index=idx))
    s = R.max_drawdown(r)
    assert s.max_drawdown == pytest.approx(0.0)
    dd = R.drawdown_series(r)
    assert (dd <= 0).all() and dd.max() == pytest.approx(0.0)


def test_drawdown_series_matches_summary():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.02, 500), index=pd.bdate_range("2020-01-01", periods=500))
    assert R.drawdown_series(r).min() == pytest.approx(R.max_drawdown(r).max_drawdown)


def test_rolling_functions_shapes():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2020-01-01", periods=300)
    a = pd.Series(rng.normal(0, 0.01, 300), index=idx)
    b = pd.Series(rng.normal(0, 0.01, 300), index=idx)
    vol = R.rolling_volatility(a, window=20)
    corr = R.rolling_correlation(a, b, window=20)
    assert vol.isna().sum() == 19 and corr.isna().sum() == 19
    assert (vol.dropna() >= 0).all()
    assert corr.dropna().between(-1, 1).all()


def test_summary_table_columns():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2020-01-01", periods=300)
    df = pd.DataFrame(rng.normal(0, 0.01, (300, 2)), index=idx, columns=["a", "b"])
    t = R.summary_table(df)
    assert list(t.columns) == ["a", "b"]
    assert "Maximum drawdown" in t.index and t.loc["Observations", "a"] == 300
