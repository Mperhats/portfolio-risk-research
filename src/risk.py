"""Tail-risk estimators: Value at Risk, Expected Shortfall, bootstrap, stress.

Sign convention
---------------
All estimates are reported as **positive losses** expressed as a fraction of
portfolio value: a 99% VaR of ``0.04`` means "an estimated loss threshold of
4% of value that historical or modelled daily returns exceeded with 1%
frequency". Multiply by the portfolio value to obtain currency losses.

Definitions (for a return ``R`` and confidence level ``alpha``):

* ``VaR_alpha = -q_{1-alpha}(R)``, the negative of the ``(1 - alpha)`` quantile.
* ``ES_alpha = -E[R | R <= q_{1-alpha}(R)]``, the average loss beyond VaR.

Expected Shortfall is a coherent risk measure (Artzner et al., 1999; Acerbi &
Tasche, 2002); VaR in general is not sub-additive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from . import config as cfg

# --------------------------------------------------------------------------- #
# Historical (empirical) estimators
# --------------------------------------------------------------------------- #


def _as_array(returns: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        raise ValueError("returns must contain at least one finite value")
    return arr


def _check_level(level: float) -> None:
    if not 0.5 < level < 1.0:
        raise ValueError(f"confidence level must be in (0.5, 1), got {level}")


def historical_var(returns: pd.Series | np.ndarray, level: float = 0.99) -> float:
    """Empirical VaR: negative of the ``(1 - level)`` sample quantile.

    Uses linear interpolation between order statistics (numpy default), so on
    a sample whose ``(1 - level)`` quantile falls exactly on an observation
    the VaR equals the negative of that observation.
    """
    _check_level(level)
    arr = _as_array(returns)
    return float(-np.quantile(arr, 1.0 - level))


def historical_es(returns: pd.Series | np.ndarray, level: float = 0.99) -> float:
    """Empirical Expected Shortfall: mean loss in the tail at or beyond VaR.

    The tail is defined as observations less than or equal to the
    ``(1 - level)`` quantile. This is always at least as large as the
    historical VaR at the same level.
    """
    _check_level(level)
    arr = _as_array(returns)
    q = np.quantile(arr, 1.0 - level)
    tail = arr[arr <= q]
    return float(-tail.mean())


# --------------------------------------------------------------------------- #
# Parametric estimators
# --------------------------------------------------------------------------- #


def gaussian_var(returns: pd.Series | np.ndarray, level: float = 0.99) -> float:
    """Normal-distribution VaR using the sample mean and standard deviation."""
    _check_level(level)
    arr = _as_array(returns)
    mu, sd = arr.mean(), arr.std(ddof=1)
    return float(-(mu + sd * stats.norm.ppf(1.0 - level)))


def gaussian_es(returns: pd.Series | np.ndarray, level: float = 0.99) -> float:
    """Normal-distribution Expected Shortfall (closed form)."""
    _check_level(level)
    arr = _as_array(returns)
    mu, sd = arr.mean(), arr.std(ddof=1)
    z = stats.norm.ppf(1.0 - level)
    return float(-(mu - sd * stats.norm.pdf(z) / (1.0 - level)))


@dataclass(frozen=True)
class StudentTFit:
    """Maximum-likelihood Student-t parameters (location-scale form)."""

    dof: float
    loc: float
    scale: float

    @property
    def defensible(self) -> bool:
        """True when the fitted tail index implies a finite variance."""
        return self.dof > cfg.T_FIT_MIN_DOF


def fit_student_t(returns: pd.Series | np.ndarray) -> StudentTFit:
    """Fit a location-scale Student-t by maximum likelihood."""
    arr = _as_array(returns)
    dof, loc, scale = stats.t.fit(arr)
    return StudentTFit(float(dof), float(loc), float(scale))


def student_t_var(returns: pd.Series | np.ndarray, level: float = 0.99, fit: StudentTFit | None = None) -> float:
    """Student-t VaR from a fitted (or supplied) location-scale t distribution."""
    _check_level(level)
    fit = fit or fit_student_t(returns)
    return float(-stats.t.ppf(1.0 - level, fit.dof, loc=fit.loc, scale=fit.scale))


def student_t_es(returns: pd.Series | np.ndarray, level: float = 0.99, fit: StudentTFit | None = None) -> float:
    """Student-t Expected Shortfall (closed form, McNeil et al. 2015, eq. 2.26).

    For the standardised t with ``nu`` degrees of freedom and lower tail
    probability ``p = 1 - level``::

        ES = (f(q_p) / p) * (nu + q_p^2) / (nu - 1)

    which is then relocated and rescaled. Requires ``nu > 1``.
    """
    _check_level(level)
    fit = fit or fit_student_t(returns)
    if fit.dof <= 1:
        return float("nan")
    p = 1.0 - level
    q = stats.t.ppf(p, fit.dof)
    es_std = stats.t.pdf(q, fit.dof) / p * (fit.dof + q**2) / (fit.dof - 1.0)
    return float(-(fit.loc - fit.scale * es_std))


# --------------------------------------------------------------------------- #
# Comparison table
# --------------------------------------------------------------------------- #


def risk_table(returns: pd.Series, levels: tuple[float, ...] = cfg.CONFIDENCE_LEVELS) -> pd.DataFrame:
    """Long table of VaR and ES by estimator and confidence level.

    Columns: ``level, measure, method, value`` where ``value`` is a positive
    loss fraction of portfolio value.
    """
    fit = fit_student_t(returns)
    rows = []
    for lvl in levels:
        rows += [
            (lvl, "VaR", "Historical", historical_var(returns, lvl)),
            (lvl, "ES", "Historical", historical_es(returns, lvl)),
            (lvl, "VaR", "Gaussian", gaussian_var(returns, lvl)),
            (lvl, "ES", "Gaussian", gaussian_es(returns, lvl)),
            (lvl, "VaR", "Student-t", student_t_var(returns, lvl, fit)),
            (lvl, "ES", "Student-t", student_t_es(returns, lvl, fit)),
        ]
    return pd.DataFrame(rows, columns=["level", "measure", "method", "value"])


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #


def _circular_block_indices(rng: np.random.Generator, n: int, block_size: int) -> np.ndarray:
    """Indices for one circular block-bootstrap resample of length ``n``."""
    if block_size <= 1:
        return rng.integers(0, n, size=n)
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n, size=n_blocks)
    idx = (starts[:, None] + np.arange(block_size)[None, :]) % n
    return idx.ravel()[:n]


def bootstrap_var_es(
    returns: pd.Series | np.ndarray,
    level: float = 0.99,
    n_boot: int = cfg.BOOTSTRAP_SAMPLES,
    block_size: int = cfg.BOOTSTRAP_BLOCK_SIZE,
    seed: int = cfg.BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Bootstrap distribution of historical VaR and ES.

    A circular block bootstrap (Politis & Romano, 1994, in its fixed-block
    form) resamples contiguous blocks of ``block_size`` days to preserve
    short-range dependence such as volatility clustering; ``block_size=1``
    reduces to the i.i.d. bootstrap.

    Returns a ``DataFrame`` with ``n_boot`` rows and columns ``VaR`` and ``ES``.
    Results are reproducible for a fixed ``seed``.
    """
    _check_level(level)
    arr = _as_array(returns)
    rng = np.random.default_rng(seed)
    n = arr.size
    out = np.empty((n_boot, 2))
    for b in range(n_boot):
        sample = arr[_circular_block_indices(rng, n, block_size)]
        out[b, 0] = historical_var(sample, level)
        out[b, 1] = historical_es(sample, level)
    return pd.DataFrame(out, columns=["VaR", "ES"])


def bootstrap_summary(
    returns: pd.Series,
    levels: tuple[float, ...] = cfg.CONFIDENCE_LEVELS,
    ci: float = 0.90,
    **kwargs,
) -> pd.DataFrame:
    """Point estimates with percentile bootstrap intervals for VaR and ES.

    Columns: ``level, measure, estimate, lower, upper, se`` where the interval
    is the central ``ci`` percentile interval of the bootstrap distribution.
    """
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    rows = []
    for lvl in levels:
        boot = bootstrap_var_es(returns, lvl, **kwargs)
        for measure, point in (("VaR", historical_var(returns, lvl)), ("ES", historical_es(returns, lvl))):
            col = boot[measure]
            rows.append((lvl, measure, point, float(col.quantile(lo_q)), float(col.quantile(hi_q)), float(col.std(ddof=1))))
    return pd.DataFrame(rows, columns=["level", "measure", "estimate", "lower", "upper", "se"])


# --------------------------------------------------------------------------- #
# Stress scenarios
# --------------------------------------------------------------------------- #


def stress_table(
    asset_returns: pd.DataFrame,
    weights: dict[str, float] = cfg.WEIGHTS,
    scenarios: tuple[cfg.StressScenario, ...] = cfg.STRESS_SCENARIOS,
    value: float = cfg.INITIAL_VALUE,
) -> pd.DataFrame:
    """Evaluate each stress scenario on the current weights.

    The special scenario named ``"Three-sigma day, all assets"`` is filled in
    from the sample: each asset falls by three times its own historical daily
    standard deviation. A "Worst historical day" row is appended using the
    realised asset returns on the portfolio's worst day.

    Columns: ``scenario, description, portfolio_return, loss_value``.
    """
    rows = []
    for sc in scenarios:
        shocks = dict(sc.shocks)
        if sc.name.startswith("Three-sigma"):
            shocks = {a: float(-3.0 * asset_returns[a].std(ddof=1)) for a in weights}
        ret = float(sum(weights[a] * shocks.get(a, 0.0) for a in weights))
        rows.append((sc.name, sc.description, ret, -ret * value))

    port = sum(asset_returns[a] * w for a, w in weights.items())
    worst_day = port.idxmin()
    worst_ret = float(port.loc[worst_day])
    rows.append(
        (
            "Worst historical day",
            f"Realised asset returns on {worst_day.date()} applied to current weights.",
            worst_ret,
            -worst_ret * value,
        )
    )
    return pd.DataFrame(rows, columns=["scenario", "description", "portfolio_return", "loss_value"])
