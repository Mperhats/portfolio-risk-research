"""Synthetic price generation and price loading.

The default dataset is generated here from :mod:`src.config` and is fully
deterministic given the seed. A user may substitute their own wide CSV of
adjusted prices; see :func:`load_prices`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg

REQUIRED_COLUMNS: tuple[str, ...] = ("date",) + cfg.ALL_SERIES


def _business_days(start: str, n_years: int) -> pd.DatetimeIndex:
    """Return approximately ``n_years`` of weekday dates starting at ``start``."""
    n_days = int(round(n_years * cfg.TRADING_DAYS_PER_YEAR))
    return pd.bdate_range(start=start, periods=n_days)


def _standardised_t(rng: np.random.Generator, dof: float, size: tuple[int, int]) -> np.ndarray:
    """Draw Student-t innovations rescaled to unit variance.

    For ``dof > 2`` the variance of a t variable is ``dof / (dof - 2)``; dividing
    by its square root keeps the *volatility* interpretation of ``ANNUAL_VOL``
    intact while making the tails heavier than Gaussian.
    """
    if dof <= 2:
        raise ValueError("Student-t dof must exceed 2 for a finite variance.")
    z = rng.standard_t(dof, size=size)
    return z / np.sqrt(dof / (dof - 2.0))


def generate_synthetic_prices(
    seed: int = cfg.SEED,
    start: str = cfg.START_DATE,
    n_years: int = cfg.N_YEARS,
    initial_price: float = 100.0,
) -> pd.DataFrame:
    """Generate a deterministic synthetic wide table of adjusted prices.

    The data-generating process is, per trading day *t* and series *i*:

    .. math::

        r_{i,t} = \\frac{\\mu_i}{252} + m_t \\frac{\\sigma_i}{\\sqrt{252}}
                  (L z_t)_i + J_{i,t}

    where ``z_t`` are i.i.d. standardised Student-t innovations, ``L`` is the
    Cholesky factor of the configured correlation matrix, ``m_t`` is a
    volatility multiplier that is greater than one inside the high-volatility
    regime, and ``J`` is an occasional negative jump with a common and an
    idiosyncratic component. Prices are ``exp(cumsum(r))`` scaled to
    ``initial_price``.

    Parameters
    ----------
    seed : Random seed for the generator.
    start : First trading date (ISO string).
    n_years : Approximate number of years of business days.
    initial_price : Starting level for every series.

    Returns
    -------
    pandas.DataFrame
        Columns ``date, asset_a, asset_b, asset_c, asset_d, benchmark``.
    """
    rng = np.random.default_rng(seed)
    dates = _business_days(start, n_years)
    n, k = len(dates), len(cfg.ALL_SERIES)

    mu = np.array([cfg.ANNUAL_MEAN[s] for s in cfg.ALL_SERIES]) / cfg.TRADING_DAYS_PER_YEAR
    sigma = np.array([cfg.ANNUAL_VOL[s] for s in cfg.ALL_SERIES]) / np.sqrt(cfg.TRADING_DAYS_PER_YEAR)

    chol = np.linalg.cholesky(cfg.CORRELATION)
    z = _standardised_t(rng, cfg.T_DOF, (n, k)) @ chol.T

    regime = np.ones(n)
    in_regime = (dates >= pd.Timestamp(cfg.HIGH_VOL_START)) & (dates <= pd.Timestamp(cfg.HIGH_VOL_END))
    regime[in_regime] = cfg.HIGH_VOL_MULTIPLIER

    # Occasional negative shocks: a common component shared by all series plus
    # an idiosyncratic component, so that jump days cluster across assets.
    jump_days = rng.random(n) < cfg.JUMP_PROBABILITY
    common_jump = rng.uniform(cfg.JUMP_MIN, cfg.JUMP_MAX, size=n) * jump_days
    idio_jump = rng.uniform(cfg.JUMP_MIN, cfg.JUMP_MAX, size=(n, k)) * jump_days[:, None]
    jumps = cfg.JUMP_COMMON_SHARE * common_jump[:, None] + (1 - cfg.JUMP_COMMON_SHARE) * idio_jump
    # Scale idiosyncratic exposure by relative volatility so riskier assets jump more.
    jumps = jumps * (sigma / sigma.mean())[None, :]

    # Drift compensation: the jump component has a negative expected value
    # (probability x mean jump size); adding it back keeps the *expected* log
    # drift equal to ANNUAL_MEAN so that the jumps change the tails, not the
    # long-run mean. Realised sample means still vary with the seed.
    expected_jump = cfg.JUMP_PROBABILITY * 0.5 * (cfg.JUMP_MIN + cfg.JUMP_MAX) * (sigma / sigma.mean())
    log_returns = (mu - expected_jump)[None, :] + regime[:, None] * sigma[None, :] * z + jumps

    log_prices = np.log(initial_price) + np.cumsum(log_returns, axis=0)
    prices = np.exp(log_prices)

    frame = pd.DataFrame(prices, columns=list(cfg.ALL_SERIES))
    frame.insert(0, "date", dates)
    return frame


def write_synthetic_prices(path: Path = cfg.SYNTHETIC_PRICES_PATH, **kwargs) -> Path:
    """Generate the synthetic dataset and write it as CSV. Returns the path."""
    frame = generate_synthetic_prices(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, date_format="%Y-%m-%d", float_format="%.6f")
    return path


def validate_prices(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate a wide price table and return it indexed by ``date``.

    Raises ``ValueError`` on missing columns, non-positive prices, missing
    values, or unsorted dates.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Price file is missing required columns: {missing}")
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.set_index("date")[list(cfg.ALL_SERIES)].astype(float)
    if not out.index.is_monotonic_increasing:
        raise ValueError("Dates must be sorted in ascending order.")
    if out.index.has_duplicates:
        raise ValueError("Duplicate dates found.")
    if out.isna().any().any():
        raise ValueError("Price table contains missing values; forward-fill before loading.")
    if (out <= 0).any().any():
        raise ValueError("All prices must be strictly positive.")
    return out


def resolve_price_path(path: Path | str | None = None) -> tuple[Path, str]:
    """Decide which price file to use.

    Order of preference:

    1. an explicit ``path`` argument;
    2. the git-ignored user file ``data/raw/adjusted_prices.csv`` if it exists;
    3. the synthetic default ``data/raw/synthetic_adjusted_prices.csv``.

    Returns the path and a short source label (``"user"`` or ``"synthetic"``).
    """
    if path is not None:
        p = Path(path)
        label = "synthetic" if p.resolve() == cfg.SYNTHETIC_PRICES_PATH.resolve() else "user"
        return p, label
    if cfg.USER_PRICES_PATH.exists():
        return cfg.USER_PRICES_PATH, "user"
    return cfg.SYNTHETIC_PRICES_PATH, "synthetic"


def load_prices(path: Path | str | None = None, regenerate_if_missing: bool = True) -> pd.DataFrame:
    """Load a wide table of adjusted prices indexed by date.

    By default the project's synthetic dataset is used. To analyse your own
    data instead, save a CSV with the columns
    ``date, asset_a, asset_b, asset_c, asset_d, benchmark`` to
    ``data/raw/adjusted_prices.csv`` (see ``data/raw/README.md``) or pass its
    path explicitly.

    Parameters
    ----------
    path : Optional explicit CSV path.
    regenerate_if_missing : If the synthetic file is requested but absent,
        generate it on the fly so the report is always renderable.
    """
    resolved, label = resolve_price_path(path)
    if not resolved.exists():
        if label == "synthetic" and regenerate_if_missing:
            write_synthetic_prices(resolved)
        else:
            raise FileNotFoundError(f"Price file not found: {resolved}")
    frame = pd.read_csv(resolved)
    prices = validate_prices(frame)
    prices.attrs["source"] = label
    prices.attrs["path"] = str(resolved)
    return prices
