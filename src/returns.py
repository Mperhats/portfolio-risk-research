"""Return, performance, and drawdown calculations.

Conventions
-----------
* ``prices`` is a wide ``DataFrame`` indexed by date, one column per series.
* Simple returns are ``P_t / P_{t-1} - 1``; log returns are ``ln(P_t / P_{t-1})``.
* Portfolio returns assume constant weights (daily rebalancing) and are the
  weighted sum of *simple* asset returns; the portfolio log return is then
  ``ln(1 + r_p)``.
* Annualisation uses ``TRADING_DAYS_PER_YEAR`` from :mod:`src.config`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from . import config as cfg


def simple_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Arithmetic (simple) returns, first observation dropped."""
    return prices.pct_change().iloc[1:]


def log_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Continuously compounded (log) returns, first observation dropped."""
    return np.log(prices).diff().iloc[1:]


def portfolio_returns(asset_returns: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Constant-weight portfolio simple return: ``r_p = sum_i w_i r_i``.

    Parameters
    ----------
    asset_returns : Simple returns, one column per asset.
    weights : Mapping asset -> weight. Must sum to one and cover only columns
        that exist in ``asset_returns``.
    """
    missing = set(weights) - set(asset_returns.columns)
    if missing:
        raise KeyError(f"weights refer to unknown columns: {sorted(missing)}")
    total = sum(weights.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"weights must sum to one, got {total:.6f}")
    w = pd.Series(weights, dtype=float)
    out = asset_returns[w.index].mul(w, axis=1).sum(axis=1)
    out.name = "portfolio"
    return out


def cumulative_growth(returns: pd.Series | pd.DataFrame, start: float = 1.0) -> pd.Series | pd.DataFrame:
    """Growth of ``start`` invested at the beginning: ``start * prod(1 + r)``."""
    return start * (1.0 + returns).cumprod()


def cumulative_return(returns: pd.Series) -> float:
    """Total simple return over the whole sample."""
    return float(np.prod(1.0 + returns.to_numpy()) - 1.0)


def annualized_mean(returns: pd.Series, periods: int = cfg.TRADING_DAYS_PER_YEAR) -> float:
    """Arithmetic annualised mean of periodic returns (``mean * periods``)."""
    return float(returns.mean() * periods)


def annualized_volatility(returns: pd.Series, periods: int = cfg.TRADING_DAYS_PER_YEAR) -> float:
    """Annualised standard deviation (``std * sqrt(periods)``, sample std)."""
    return float(returns.std(ddof=1) * np.sqrt(periods))


def skewness(returns: pd.Series) -> float:
    """Sample skewness (bias-corrected, as in ``pandas.Series.skew``)."""
    return float(stats.skew(returns.to_numpy(), bias=False))


def excess_kurtosis(returns: pd.Series) -> float:
    """Sample excess kurtosis (zero for a normal distribution)."""
    return float(stats.kurtosis(returns.to_numpy(), fisher=True, bias=False))


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Drawdown from the running peak of cumulative growth (values <= 0)."""
    growth = cumulative_growth(returns)
    peak = growth.cummax()
    dd = growth / peak - 1.0
    dd.name = "drawdown"
    return dd


@dataclass(frozen=True)
class DrawdownSummary:
    """Maximum drawdown and its timing."""

    max_drawdown: float
    peak_date: pd.Timestamp
    trough_date: pd.Timestamp
    recovery_date: pd.Timestamp | None
    duration_days: int  # trading days from peak to trough


def max_drawdown(returns: pd.Series) -> DrawdownSummary:
    """Compute the maximum peak-to-trough decline of cumulative growth.

    The drawdown is reported as a negative fraction (e.g. ``-0.35``). The
    recovery date is the first date after the trough at which the running
    peak is regained, or ``None`` if it never is within the sample.
    """
    growth = cumulative_growth(returns)
    peak = growth.cummax()
    dd = growth / peak - 1.0
    trough = dd.idxmin()
    mdd = float(dd.loc[trough])
    prior = growth.loc[:trough]
    peak_date = prior.idxmax()
    after = growth.loc[trough:]
    recovered = after[after >= growth.loc[peak_date]]
    recovery = recovered.index[0] if len(recovered) else None
    duration = int(growth.index.get_loc(trough) - growth.index.get_loc(peak_date))
    return DrawdownSummary(mdd, peak_date, trough, recovery, duration)


def rolling_volatility(
    returns: pd.Series, window: int = cfg.ROLLING_WINDOW, periods: int = cfg.TRADING_DAYS_PER_YEAR
) -> pd.Series:
    """Rolling annualised volatility over ``window`` observations."""
    out = returns.rolling(window).std(ddof=1) * np.sqrt(periods)
    out.name = f"rolling_vol_{window}"
    return out


def rolling_correlation(a: pd.Series, b: pd.Series, window: int = cfg.ROLLING_WINDOW) -> pd.Series:
    """Rolling Pearson correlation between two return series."""
    out = a.rolling(window).corr(b)
    out.name = f"rolling_corr_{window}"
    return out


def summary_table(returns: pd.DataFrame, periods: int = cfg.TRADING_DAYS_PER_YEAR) -> pd.DataFrame:
    """Per-column performance and distribution summary.

    Rows: annualised mean, annualised volatility, cumulative return, skewness,
    excess kurtosis, maximum drawdown, worst day, best day, observations.
    """
    rows = {}
    for col in returns.columns:
        r = returns[col].dropna()
        rows[col] = {
            "Annualised mean": annualized_mean(r, periods),
            "Annualised volatility": annualized_volatility(r, periods),
            "Cumulative return": cumulative_return(r),
            "Skewness": skewness(r),
            "Excess kurtosis": excess_kurtosis(r),
            "Maximum drawdown": max_drawdown(r).max_drawdown,
            "Worst day": float(r.min()),
            "Best day": float(r.max()),
            "Observations": int(r.shape[0]),
        }
    return pd.DataFrame(rows)
