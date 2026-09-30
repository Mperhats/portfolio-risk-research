"""Tests for src.simulation."""

import numpy as np
import pandas as pd
import pytest

from src import simulation as S


@pytest.fixture
def log_r() -> pd.Series:
    rng = np.random.default_rng(9)
    return pd.Series(rng.standard_t(5, 1000) * 0.008 + 0.0003)


@pytest.mark.parametrize("engine", [S.simulate_lognormal, S.simulate_student_t, S.simulate_bootstrap])
def test_shapes_and_start_value(engine, log_r):
    res = engine(log_r, n_paths=200, horizon=30, v0=1000.0, seed=1)
    assert res.paths.shape == (200, 31)
    assert np.allclose(res.paths[:, 0], 1000.0)
    assert res.terminal.shape == (200,)
    assert (res.paths > 0).all()


@pytest.mark.parametrize("engine", [S.simulate_lognormal, S.simulate_student_t, S.simulate_bootstrap])
def test_reproducible_with_seed(engine, log_r):
    a = engine(log_r, n_paths=100, horizon=20, seed=42)
    b = engine(log_r, n_paths=100, horizon=20, seed=42)
    c = engine(log_r, n_paths=100, horizon=20, seed=43)
    np.testing.assert_array_equal(a.paths, b.paths)
    assert not np.array_equal(a.paths, c.paths)


def test_lognormal_mean_matches_theory():
    # With mu_d, sd_d per day, E[V_T] = V0 * exp(T * (mu_d + sd_d^2 / 2)).
    rng = np.random.default_rng(10)
    r = pd.Series(rng.normal(0.0004, 0.01, 5000))
    res = S.simulate_lognormal(r, n_paths=40000, horizon=252, v0=1.0, seed=3)
    mu, sd = r.mean(), r.std(ddof=1)
    expected = np.exp(252 * (mu + sd**2 / 2))
    assert res.terminal.mean() == pytest.approx(expected, rel=0.03)


def test_bootstrap_only_uses_observed_returns(log_r):
    res = S.simulate_bootstrap(log_r, n_paths=50, horizon=40, v0=1.0, seed=2, block_size=5)
    # Recover the daily log returns from the paths (exp/log round-trip => ~1e-9 noise)
    steps = np.diff(np.log(res.paths), axis=1).ravel()
    observed = log_r.to_numpy()
    nearest = np.abs(steps[:, None] - observed[None, :]).min(axis=1)
    assert nearest.max() < 1e-8


def test_quantiles_fan_and_probabilities(log_r):
    res = S.simulate_student_t(log_r, n_paths=500, horizon=60, v0=100.0, seed=5)
    q = res.quantiles((0.05, 0.5, 0.95))
    assert q[0.05] < q[0.5] < q[0.95]
    fan = res.fan((0.05, 0.5, 0.95))
    assert fan.shape == (61, 3)
    assert (fan[0.05] <= fan[0.5]).all() and (fan[0.5] <= fan[0.95]).all()
    p10, p50 = res.prob_loss_exceeds(0.10), res.prob_loss_exceeds(0.50)
    assert 0.0 <= p50 <= p10 <= 1.0


def test_run_all_and_summary(log_r):
    results = S.run_all(log_r, n_paths=300, horizon=20, seed=4)
    assert set(results) == {"Normal (lognormal)", "Student-t", "Block bootstrap"}
    t = S.summary_table(results)
    assert "Median" in t.index and "P(loss > 10%)" in t.index
    assert t.shape[1] == 3
