"""Central configuration: every modelling assumption lives here.

Nothing in this file refers to a real security. The portfolio is an
illustrative, deliberately concentrated allocation used to study tail-risk
estimators on synthetic data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_RAW: Path = PROJECT_ROOT / "data" / "raw"
DATA_DERIVED: Path = PROJECT_ROOT / "data" / "derived"

SYNTHETIC_PRICES_PATH: Path = DATA_RAW / "synthetic_adjusted_prices.csv"
#: Optional user-supplied file in the same wide format. If it exists it is
#: preferred by :func:`src.data.load_prices`; it is git-ignored.
USER_PRICES_PATH: Path = DATA_RAW / "adjusted_prices.csv"

# --------------------------------------------------------------------------- #
# Universe and portfolio
# --------------------------------------------------------------------------- #
ASSETS: tuple[str, ...] = ("asset_a", "asset_b", "asset_c", "asset_d")
BENCHMARK: str = "benchmark"
ALL_SERIES: tuple[str, ...] = ASSETS + (BENCHMARK,)

#: Display labels used in tables and figures.
LABELS: dict[str, str] = {
    "asset_a": "Asset A",
    "asset_b": "Asset B",
    "asset_c": "Asset C",
    "asset_d": "Asset D",
    "benchmark": "Benchmark",
    "portfolio": "Portfolio",
}

#: Fixed target weights. The analysis assumes daily rebalancing back to these
#: weights, which is a simplification stated in the report.
WEIGHTS: dict[str, float] = {
    "asset_a": 0.45,
    "asset_b": 0.25,
    "asset_c": 0.20,
    "asset_d": 0.10,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-12, "weights must sum to one"

#: Illustrative notional used to express losses in currency units.
INITIAL_VALUE: float = 1_000_000.0
CURRENCY: str = "USD"

TRADING_DAYS_PER_YEAR: int = 252

# --------------------------------------------------------------------------- #
# Synthetic data-generating process
# --------------------------------------------------------------------------- #
#: Chosen (from a handful of candidate seeds) because the resulting path is
#: representative rather than degenerate: a deep but recoverable drawdown,
#: visibly fat tails, and neither a runaway nor a collapsing benchmark.
SEED: int = 20240615
START_DATE: str = "2018-01-02"
N_YEARS: int = 6  # ~6 years of business days

#: Annualised drift of *log* returns and annualised volatility, per series.
ANNUAL_MEAN: dict[str, float] = {
    "asset_a": 0.11,
    "asset_b": 0.07,
    "asset_c": 0.05,
    "asset_d": 0.09,
    "benchmark": 0.075,
}
ANNUAL_VOL: dict[str, float] = {
    "asset_a": 0.30,
    "asset_b": 0.22,
    "asset_c": 0.17,
    "asset_d": 0.36,
    "benchmark": 0.16,
}

#: Correlation of daily innovations (order = ALL_SERIES).
CORRELATION: np.ndarray = np.array(
    [
        # a      b      c      d      bench
        [1.00, 0.55, 0.35, 0.45, 0.80],
        [0.55, 1.00, 0.40, 0.30, 0.70],
        [0.35, 0.40, 1.00, 0.20, 0.55],
        [0.45, 0.30, 0.20, 1.00, 0.60],
        [0.80, 0.70, 0.55, 0.60, 1.00],
    ]
)

#: Degrees of freedom of the Student-t innovations (lower = fatter tails).
T_DOF: float = 5.0

#: High-volatility regime: volatility multiplier applied between two dates.
HIGH_VOL_START: str = "2020-02-20"
HIGH_VOL_END: str = "2020-06-30"
HIGH_VOL_MULTIPLIER: float = 2.2

#: Occasional negative shocks: daily probability and shock size distribution
#: (log-return, drawn uniformly in [JUMP_MIN, JUMP_MAX], always negative).
JUMP_PROBABILITY: float = 0.004
JUMP_MIN: float = -0.07
JUMP_MAX: float = -0.02
#: Fraction of a jump that is common to all series (the rest is idiosyncratic).
JUMP_COMMON_SHARE: float = 0.6

# --------------------------------------------------------------------------- #
# Risk-measure settings
# --------------------------------------------------------------------------- #
CONFIDENCE_LEVELS: tuple[float, ...] = (0.95, 0.99)
ROLLING_WINDOW: int = 63  # ~ one quarter of trading days

BOOTSTRAP_SAMPLES: int = 2_000
BOOTSTRAP_BLOCK_SIZE: int = 10  # circular block bootstrap; 1 = i.i.d.
BOOTSTRAP_SEED: int = 7

# Minimum degrees of freedom accepted for the Student-t fit; below this the
# variance is undefined and the parametric VaR is not reported.
T_FIT_MIN_DOF: float = 2.05

# --------------------------------------------------------------------------- #
# Simulation settings
# --------------------------------------------------------------------------- #
SIM_PATHS: int = 20_000
SIM_HORIZON_DAYS: int = TRADING_DAYS_PER_YEAR
SIM_SEED: int = 11
SIM_QUANTILES: tuple[float, ...] = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)

# --------------------------------------------------------------------------- #
# Stress scenarios
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StressScenario:
    """An instantaneous, hypothetical shock applied to each asset.

    ``shocks`` maps asset name to a simple return (e.g. ``-0.20`` = a 20% fall).
    Assets that are not listed are assumed unchanged. Shocks are applied to the
    *current* weights with no rebalancing, hedging, or liquidity effects.
    """

    name: str
    description: str
    shocks: dict[str, float] = field(default_factory=dict)

    def portfolio_return(self, weights: dict[str, float]) -> float:
        """Return the portfolio simple return implied by the shock."""
        return float(sum(weights[a] * self.shocks.get(a, 0.0) for a in weights))


STRESS_SCENARIOS: tuple[StressScenario, ...] = (
    StressScenario(
        name="Broad market fall, 20%",
        description="Every asset falls 20% at once; correlations go to one.",
        shocks={a: -0.20 for a in ASSETS},
    ),
    StressScenario(
        name="Largest holding halves",
        description="Idiosyncratic 50% fall in Asset A; other assets unchanged.",
        shocks={"asset_a": -0.50},
    ),
    StressScenario(
        name="Concentration shock",
        description="Assets A and D (the two most volatile) fall 35%; B and C fall 10%.",
        shocks={"asset_a": -0.35, "asset_d": -0.35, "asset_b": -0.10, "asset_c": -0.10},
    ),
    StressScenario(
        name="Defensive assets fail",
        description="The lower-volatility sleeve (B and C) falls 25%; A and D fall 5%.",
        shocks={"asset_b": -0.25, "asset_c": -0.25, "asset_a": -0.05, "asset_d": -0.05},
    ),
    StressScenario(
        name="Three-sigma day, all assets",
        description="Each asset falls by three times its own historical daily volatility.",
        shocks={},  # filled at runtime from the sample; see risk.stress_table
    ),
)
