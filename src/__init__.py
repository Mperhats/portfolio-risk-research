"""Analysis package for the portfolio-risk research memo.

Modules
-------
config      : all assumptions (weights, seeds, horizons, scenario definitions)
data        : synthetic price generation and CSV loading
returns     : return, performance and drawdown calculations
risk        : VaR / Expected Shortfall estimators, bootstrap, stress tests
simulation  : one-year Monte Carlo terminal-value simulations
figures     : Tufte-inspired matplotlib figures used by the Quarto report

Import submodules explicitly, e.g. ``from src import risk``; the package
itself deliberately imports nothing so that a heavy plotting dependency is
only loaded when a figure is actually requested.
"""

__all__ = ["config", "data", "figures", "returns", "risk", "simulation"]
