"""One-year Monte Carlo simulations of portfolio value.

Three engines share one interface and return a ``SimulationResult``:

``simulate_lognormal``
    i.i.d. Gaussian daily log returns with the sample mean and variance.
    Terminal value is therefore lognormal.
``simulate_student_t``
    i.i.d. daily log returns from a fitted location-scale Student-t.
``simulate_bootstrap``
    circular block bootstrap of the historical daily log returns.

All engines are deterministic for a fixed ``seed``. They model the *portfolio*
return series directly (constant weights), ignore transaction costs and
rebalancing frictions, and assume the historical sample is representative of
the horizon, which is an assumption and not a forecast.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from . import config as cfg
from .risk import StudentTFit, fit_student_t


@dataclass(frozen=True)
class SimulationResult:
    """Simulated wealth paths and derived summaries."""

    method: str
    paths: np.ndarray  # shape (n_paths, horizon + 1), paths[:, 0] == v0
    v0: float

    @property
    def terminal(self) -> np.ndarray:
        """Terminal portfolio values, shape ``(n_paths,)``."""
        return self.paths[:, -1]

    @property
    def terminal_return(self) -> np.ndarray:
        """Simple return over the horizon, shape ``(n_paths,)``."""
        return self.terminal / self.v0 - 1.0

    def quantiles(self, qs: tuple[float, ...] = cfg.SIM_QUANTILES) -> pd.Series:
        """Terminal-value quantiles indexed by probability."""
        return pd.Series(np.quantile(self.terminal, qs), index=list(qs), name=self.method)

    def fan(self, qs: tuple[float, ...] = cfg.SIM_QUANTILES) -> pd.DataFrame:
        """Path quantiles through time (rows = day, columns = probability)."""
        q = np.quantile(self.paths, qs, axis=0).T
        return pd.DataFrame(q, columns=list(qs))

    def prob_loss_exceeds(self, threshold: float) -> float:
        """Probability that the horizon loss exceeds ``threshold`` (fraction)."""
        return float(np.mean(self.terminal_return <= -threshold))


def _paths_from_log_returns(log_r: np.ndarray, v0: float) -> np.ndarray:
    """Wealth paths from a matrix of daily log returns (n_paths, horizon)."""
    cum = np.cumsum(log_r, axis=1)
    paths = v0 * np.exp(cum)
    return np.concatenate([np.full((log_r.shape[0], 1), v0), paths], axis=1)


def simulate_lognormal(
    log_returns: pd.Series | np.ndarray,
    n_paths: int = cfg.SIM_PATHS,
    horizon: int = cfg.SIM_HORIZON_DAYS,
    v0: float = cfg.INITIAL_VALUE,
    seed: int = cfg.SIM_SEED,
) -> SimulationResult:
    """Gaussian i.i.d. daily log-return simulation (lognormal terminal value).

    The terminal value is ``V_T = V_0 exp(sum_t r_t)`` with
    ``r_t ~ N(mu_d, sigma_d^2)`` estimated from the sample.
    """
    arr = np.asarray(log_returns, dtype=float)
    mu, sd = arr.mean(), arr.std(ddof=1)
    rng = np.random.default_rng(seed)
    log_r = rng.normal(mu, sd, size=(n_paths, horizon))
    return SimulationResult("Normal (lognormal)", _paths_from_log_returns(log_r, v0), v0)


def simulate_student_t(
    log_returns: pd.Series | np.ndarray,
    n_paths: int = cfg.SIM_PATHS,
    horizon: int = cfg.SIM_HORIZON_DAYS,
    v0: float = cfg.INITIAL_VALUE,
    seed: int = cfg.SIM_SEED,
    fit: StudentTFit | None = None,
    recenter: bool = True,
) -> SimulationResult:
    """i.i.d. daily log returns drawn from a fitted location-scale Student-t.

    The maximum-likelihood location of a Student-t is a *robust* estimate that
    down-weights extreme days; on a sample with negative jumps it sits above
    the arithmetic sample mean. With ``recenter=True`` (default) the location
    is shifted so that the simulated mean daily log return equals the sample
    mean, which makes the engine comparable with the lognormal baseline: the
    two then share drift and differ only in the shape of the daily
    distribution. ``recenter=False`` uses the raw MLE location.
    """
    arr = np.asarray(log_returns, dtype=float)
    fit = fit or fit_student_t(arr)
    loc = float(arr.mean()) if recenter else fit.loc  # mean of a t with dof > 1 is its location
    rng = np.random.default_rng(seed)
    log_r = stats.t.rvs(fit.dof, loc=loc, scale=fit.scale, size=(n_paths, horizon), random_state=rng)
    return SimulationResult("Student-t", _paths_from_log_returns(log_r, v0), v0)


def simulate_bootstrap(
    log_returns: pd.Series | np.ndarray,
    n_paths: int = cfg.SIM_PATHS,
    horizon: int = cfg.SIM_HORIZON_DAYS,
    v0: float = cfg.INITIAL_VALUE,
    seed: int = cfg.SIM_SEED,
    block_size: int = cfg.BOOTSTRAP_BLOCK_SIZE,
) -> SimulationResult:
    """Circular block bootstrap of historical daily log returns.

    Each path concatenates randomly chosen contiguous blocks of the historical
    series until ``horizon`` days are filled, preserving short-range
    volatility clustering. No distributional assumption is imposed, but the
    simulation cannot produce days worse than the worst historical day.
    """
    arr = np.asarray(log_returns, dtype=float)
    rng = np.random.default_rng(seed)
    n = arr.size
    block_size = max(1, int(block_size))
    n_blocks = int(np.ceil(horizon / block_size))
    # Vectorised: one matrix of block start positions for every path at once.
    starts = rng.integers(0, n, size=(n_paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(block_size)[None, None, :]) % n
    idx = idx.reshape(n_paths, -1)[:, :horizon]
    log_r = arr[idx]
    return SimulationResult("Block bootstrap", _paths_from_log_returns(log_r, v0), v0)


def run_all(
    log_returns: pd.Series,
    n_paths: int = cfg.SIM_PATHS,
    horizon: int = cfg.SIM_HORIZON_DAYS,
    v0: float = cfg.INITIAL_VALUE,
    seed: int = cfg.SIM_SEED,
) -> dict[str, SimulationResult]:
    """Run the three engines with a common seed and return them by method name."""
    results = [
        simulate_lognormal(log_returns, n_paths, horizon, v0, seed),
        simulate_student_t(log_returns, n_paths, horizon, v0, seed),
        simulate_bootstrap(log_returns, n_paths, horizon, v0, seed),
    ]
    return {r.method: r for r in results}


def _ordinal(n: int) -> str:
    """1 -> '1st', 2 -> '2nd', 25 -> '25th'."""
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def summary_table(results: dict[str, SimulationResult], thresholds: tuple[float, ...] = (0.10, 0.20, 0.30)) -> pd.DataFrame:
    """Compare engines: terminal quantiles and tail-loss probabilities.

    Rows are statistics, columns are methods. Values are terminal values in
    currency for the quantile rows and probabilities for the loss rows.
    """
    frames = {}
    for name, res in results.items():
        q = res.quantiles()
        row = {_ordinal(int(round(p * 100))) + " pct" if p != 0.5 else "Median": v for p, v in q.items()}
        row["Mean"] = float(res.terminal.mean())
        for th in thresholds:
            row[f"P(loss > {int(th*100)}%)"] = res.prob_loss_exceeds(th)
        frames[name] = row
    return pd.DataFrame(frames)
