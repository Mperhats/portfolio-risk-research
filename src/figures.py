"""Tufte-inspired matplotlib figures for the research memo.

Design rules applied throughout (see the report's design note):

* one restrained palette: deep blue for the primary series, charcoal ink for
  text and secondary series, muted grays for context, rust **only** for losses
  and tail emphasis;
* thin marks, hairline solid gridlines, no top/right spines, no chart junk;
* direct labels at the end of lines instead of legend boxes where possible;
* units on axes, percentages formatted consistently, currency in thousands.

Every function returns a ``matplotlib.figure.Figure`` and does not call
``plt.show()``; Quarto's inline backend displays the figure at the end of
the cell.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, PercentFormatter
from scipy import stats

from . import config as cfg
from .risk import StudentTFit
from .simulation import SimulationResult

# --------------------------------------------------------------------------- #
# Palette and style
# --------------------------------------------------------------------------- #
BLUE = "#25609f"  # primary data
BLUE_MID = "#7e9fc7"  # secondary blue tint (bands)
BLUE_LIGHT = "#c6d5e8"  # lightest blue tint (outer bands)
INK = "#2b2f36"  # charcoal for text and secondary lines
GRAY = "#9aa0a6"  # context series
GRAY_LIGHT = "#d9dcdf"  # gridlines, reference lines
GRAY_FILL = "#eceef0"  # regime shading, histogram fills
RUST = "#b04a2f"  # loss / tail emphasis only
RUST_LIGHT = "#e6c3b8"  # tail fills
TEXT = "#3a3d42"
TEXT_MUTED = "#6c7075"

PALETTE = {"blue": BLUE, "ink": INK, "gray": GRAY, "rust": RUST}

_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "normal",
    "axes.titlelocation": "left",
    "axes.labelsize": 9,
    "axes.labelcolor": TEXT,
    "axes.edgecolor": GRAY_LIGHT,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": GRAY_LIGHT,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "axes.prop_cycle": mpl.cycler(color=[BLUE, INK, GRAY, RUST]),
    "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.major.size": 3,
    "ytick.major.size": 0,
    "xtick.major.width": 0.6,
    "xtick.direction": "out",
    "legend.frameon": False,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "text.color": TEXT,
    "lines.linewidth": 1.4,
    "lines.solid_capstyle": "round",
    "svg.fonttype": "path",
}


def apply_style() -> None:
    """Apply the memo's matplotlib style globally (idempotent)."""
    mpl.rcParams.update(_STYLE)


def _label(s: str) -> str:
    return cfg.LABELS.get(s, s)


def _pct(decimals: int = 0) -> PercentFormatter:
    return PercentFormatter(xmax=1.0, decimals=decimals)


def _thousands(x: float, _pos=None) -> str:
    return f"{x/1000:,.0f}k"


def _annotate_end(ax, x, y, text: str, color: str, dx: float = 6, **kw) -> None:
    """Direct label just to the right of a line's last point."""
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, 0),
        textcoords="offset points",
        va="center",
        ha="left",
        fontsize=8,
        color=color,
        annotation_clip=False,
        **kw,
    )


def _shade_regime(ax, index: pd.DatetimeIndex, label: bool = True) -> None:
    """Shade the configured high-volatility regime as quiet context."""
    start, end = pd.Timestamp(cfg.HIGH_VOL_START), pd.Timestamp(cfg.HIGH_VOL_END)
    if index.min() <= end and index.max() >= start:
        ax.axvspan(max(start, index.min()), min(end, index.max()), color=GRAY_FILL, lw=0, zorder=0)
        if label:
            ax.text(
                start,
                0.02,
                " high-volatility\n regime",
                transform=ax.get_xaxis_transform(),
                fontsize=6.5,
                color=TEXT_MUTED,
                va="bottom",
                ha="left",
                linespacing=1.1,
            )


def _date_axis(ax) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.margins(x=0.01)


# --------------------------------------------------------------------------- #
# 1. Cumulative growth
# --------------------------------------------------------------------------- #


def cumulative_growth(
    growth: pd.DataFrame,
    primary: str = "portfolio",
    benchmark: str = cfg.BENCHMARK,
    context: Sequence[str] = cfg.ASSETS,
    figsize: tuple[float, float] = (10, 4.2),
) -> Figure:
    """Growth of one unit for the portfolio, the benchmark and, in light gray, the assets."""
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)
    _shade_regime(ax, growth.index)
    x_end = growth.index[-1]
    for c in context:
        ax.plot(growth.index, growth[c], color=GRAY_LIGHT, lw=0.9, zorder=1)
        _annotate_end(ax, x_end, growth[c].iloc[-1], _label(c), GRAY)
    ax.plot(growth.index, growth[benchmark], color=GRAY, lw=1.2, zorder=2)
    _annotate_end(ax, x_end, growth[benchmark].iloc[-1], _label(benchmark), TEXT_MUTED)
    ax.plot(growth.index, growth[primary], color=BLUE, lw=1.8, zorder=3)
    _annotate_end(ax, x_end, growth[primary].iloc[-1], _label(primary), BLUE, fontweight="bold")
    ax.set_yscale("log")
    ticks = [0.25, 0.5, 1, 2, 4, 8]
    lo, hi = growth.min().min(), growth.max().max()
    ticks = [t for t in ticks if lo * 0.8 <= t <= hi * 1.25] or [1]
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.axhline(1.0, color=GRAY_LIGHT, lw=0.8, zorder=0)
    ax.set_ylabel("Growth of 1 unit (log scale)")
    _date_axis(ax)
    return fig


# --------------------------------------------------------------------------- #
# 2. Return distribution with normal and Student-t overlays
# --------------------------------------------------------------------------- #


def return_distribution(
    returns: pd.Series,
    tfit: StudentTFit,
    var99: float | None = None,
    figsize: tuple[float, float] = (10, 3.8),
) -> Figure:
    """Daily-return histogram with fitted normal and Student-t densities.

    Left: linear density. Right: the same on a log scale, which is where the
    difference between the two models is visible. The region beyond the 99%
    historical loss threshold is shaded in rust on the left.
    """
    apply_style()
    r = returns.to_numpy()
    mu, sd = r.mean(), r.std(ddof=1)
    xs = np.linspace(r.min() * 1.05, r.max() * 1.05, 600)
    norm_pdf = stats.norm.pdf(xs, mu, sd)
    t_pdf = stats.t.pdf(xs, tfit.dof, loc=tfit.loc, scale=tfit.scale)
    bins = np.linspace(r.min(), r.max(), 90)

    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"wspace": 0.22})
    for i, ax in enumerate(axes):
        ax.hist(r, bins=bins, density=True, color=GRAY_FILL, edgecolor="white", lw=0.3, zorder=1)
        ax.plot(xs, norm_pdf, color=GRAY, lw=1.2, zorder=2)
        ax.plot(xs, t_pdf, color=BLUE, lw=1.6, zorder=3)
        ax.xaxis.set_major_formatter(_pct(0))
        ax.set_xlabel("Daily portfolio return")
        ax.grid(False)
        if i == 0:
            ax.set_ylabel("Density")
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            if var99 is not None:
                ax.fill_between(xs, 0, t_pdf, where=xs <= -var99, color=RUST_LIGHT, zorder=2)
                ax.axvline(-var99, color=RUST, lw=0.8)
                ax.text(-var99, ax.get_ylim()[1] * 0.97, f" 99% loss threshold\n {var99:.1%}", color=RUST, fontsize=7.5, va="top")
            peak_x = xs[np.argmax(t_pdf)]
            ax.annotate("Student-t fit", xy=(peak_x + 1.5 * sd, stats.t.pdf(peak_x + 1.5 * sd, tfit.dof, tfit.loc, tfit.scale)),
                        xytext=(28, 8), textcoords="offset points", color=BLUE, fontsize=8,
                        arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.6))
            ax.annotate("Normal", xy=(mu + 1.0 * sd, stats.norm.pdf(mu + 1.0 * sd, mu, sd)),
                        xytext=(34, 22), textcoords="offset points", color=GRAY, fontsize=8,
                        arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.6))
            ax.set_title("Linear scale", color=TEXT_MUTED)
        else:
            ax.set_yscale("log")
            ymin = max(t_pdf.min(), 1e-2)
            ax.set_ylim(ymin, t_pdf.max() * 3)
            ax.set_ylabel("Density (log scale)")
            ax.set_title("Log scale: the tails", color=TEXT_MUTED)
            ax.annotate("Student-t", xy=(xs[-1], t_pdf[-1]), xytext=(-2, 7), textcoords="offset points",
                        color=BLUE, fontsize=8, ha="right", va="bottom")
            # Label the normal where it is still on-scale (about 3.5 sd out).
            xn = mu + 3.2 * sd
            ax.annotate("Normal", xy=(xn, stats.norm.pdf(xn, mu, sd)), xytext=(10, 10), textcoords="offset points",
                        color=GRAY, fontsize=8, arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.6))
    return fig


# --------------------------------------------------------------------------- #
# 3. QQ plots
# --------------------------------------------------------------------------- #


def qq_plots(returns: pd.Series, tfit: StudentTFit, figsize: tuple[float, float] = (7.2, 3.6)) -> Figure:
    """Quantile-quantile plots against the fitted normal and Student-t."""
    apply_style()
    r = np.sort(returns.to_numpy())
    n = r.size
    probs = (np.arange(1, n + 1) - 0.5) / n
    mu, sd = r.mean(), r.std(ddof=1)
    theo = {
        "Normal": stats.norm.ppf(probs, mu, sd),
        f"Student-t (ν = {tfit.dof:.1f})": stats.t.ppf(probs, tfit.dof, loc=tfit.loc, scale=tfit.scale),
    }
    tail = int(np.ceil(0.01 * n))
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, (name, q) in zip(axes, theo.items()):
        lim = (min(q.min(), r.min()) * 1.05, max(q.max(), r.max()) * 1.05)
        ax.plot(lim, lim, color=GRAY_LIGHT, lw=0.9, zorder=1)
        ax.scatter(q, r, s=6, color=INK, alpha=0.55, lw=0, zorder=2)
        ax.scatter(q[:tail], r[:tail], s=9, color=RUST, lw=0, zorder=3)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.xaxis.set_major_formatter(_pct(0))
        ax.yaxis.set_major_formatter(_pct(0))
        ax.set_xlabel(f"{name} quantile")
        ax.grid(False)
        ax.set_title(name, color=TEXT_MUTED)
    axes[0].set_ylabel("Observed daily return")
    axes[0].text(0.03, 0.97, "worst 1% of days in rust", transform=axes[0].transAxes, color=RUST, fontsize=7.5, va="top")
    return fig


# --------------------------------------------------------------------------- #
# 4. Drawdown
# --------------------------------------------------------------------------- #


def drawdown(dd: pd.Series, summary, figsize: tuple[float, float] = (10, 3.2)) -> Figure:
    """Underwater chart: distance from the running peak, with the maximum annotated."""
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)
    _shade_regime(ax, dd.index, label=False)
    ax.fill_between(dd.index, dd, 0, color=RUST_LIGHT, lw=0, zorder=1)
    ax.plot(dd.index, dd, color=RUST, lw=0.9, zorder=2)
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.yaxis.set_major_formatter(_pct(0))
    ax.set_ylabel("Drawdown from peak")
    ax.set_ylim(min(dd.min() * 1.15, -0.05), 0.02)
    _date_axis(ax)
    t, v = summary.trough_date, summary.max_drawdown
    rec = summary.recovery_date.date() if summary.recovery_date is not None else "not within sample"
    ax.annotate(
        f"maximum drawdown {v:.1%}\npeak {summary.peak_date.date()} to trough {t.date()}\n"
        f"{summary.duration_days} trading days down; recovery: {rec}",
        xy=(t, v),
        xytext=(28, 14),
        textcoords="offset points",
        fontsize=7.5,
        color=RUST,
        va="bottom",
        arrowprops=dict(arrowstyle="-", color=RUST, lw=0.6),
    )
    return fig


# --------------------------------------------------------------------------- #
# 5. Rolling volatility and correlation
# --------------------------------------------------------------------------- #


def rolling_diagnostics(
    vol_port: pd.Series,
    vol_bench: pd.Series,
    corr: pd.Series,
    window: int = cfg.ROLLING_WINDOW,
    figsize: tuple[float, float] = (10, 4.6),
) -> Figure:
    """Two stacked panels: rolling annualised volatility and rolling correlation."""
    apply_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={"hspace": 0.18, "height_ratios": [1.3, 1]})
    for ax in (ax1, ax2):
        _shade_regime(ax, vol_port.index, label=ax is ax1)
    x_end = vol_port.index[-1]
    ax1.plot(vol_bench.index, vol_bench, color=GRAY, lw=1.1)
    ax1.plot(vol_port.index, vol_port, color=BLUE, lw=1.6)
    _annotate_end(ax1, x_end, vol_port.iloc[-1], "Portfolio", BLUE)
    _annotate_end(ax1, x_end, vol_bench.iloc[-1], "Benchmark", TEXT_MUTED)
    ax1.yaxis.set_major_formatter(_pct(0))
    ax1.set_ylabel(f"{window}-day volatility, annualised")
    ax1.set_ylim(0, None)
    ax2.plot(corr.index, corr, color=INK, lw=1.2)
    ax2.axhline(corr.mean(), color=GRAY_LIGHT, lw=0.9)
    ax2.text(corr.index[0], corr.mean(), f" full-sample mean {corr.mean():.2f}", color=TEXT_MUTED, fontsize=7.5, va="bottom")
    ax2.set_ylabel(f"{window}-day correlation\nwith benchmark")
    ax2.set_ylim(min(0, corr.min() - 0.05), 1.0)
    _date_axis(ax2)
    return fig


# --------------------------------------------------------------------------- #
# 6. VaR vs ES comparison
# --------------------------------------------------------------------------- #


def var_es_comparison(table: pd.DataFrame, figsize: tuple[float, float] = (7.2, 3.0)) -> Figure:
    """Dot plot of VaR (open blue) and ES (filled rust) by method and level."""
    apply_style()
    levels = sorted(table["level"].unique())
    methods = ["Historical", "Gaussian", "Student-t"]
    fig, axes = plt.subplots(1, len(levels), figsize=figsize, sharey=True, gridspec_kw={"wspace": 0.08})
    xmax = table["value"].max() * 1.25
    for ax, lvl in zip(np.atleast_1d(axes), levels):
        sub = table[table["level"] == lvl].pivot(index="method", columns="measure", values="value").loc[methods]
        y = np.arange(len(methods))[::-1]
        ax.hlines(y, sub["VaR"], sub["ES"], color=GRAY_LIGHT, lw=1.6, zorder=1)
        ax.scatter(sub["VaR"], y, s=42, facecolor="white", edgecolor=BLUE, lw=1.4, zorder=3)
        ax.scatter(sub["ES"], y, s=42, color=RUST, lw=0, zorder=3)
        for yi, (v, e) in zip(y, zip(sub["VaR"], sub["ES"])):
            ax.text(v, yi + 0.22, f"{v:.1%}", ha="center", va="bottom", fontsize=7.5, color=BLUE)
            ax.text(e, yi - 0.22, f"{e:.1%}", ha="center", va="top", fontsize=7.5, color=RUST)
        ax.set_yticks(y)
        ax.set_yticklabels(methods)
        ax.set_xlim(0, xmax)
        ax.xaxis.set_major_formatter(_pct(0))
        ax.set_xlabel("One-day loss, % of value")
        ax.set_title(f"{lvl:.0%} confidence", color=TEXT_MUTED)
        ax.grid(False)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.7, len(methods) - 0.3)
    axes[0].text(0.0, 1.14, "○ VaR   ● Expected Shortfall", transform=axes[0].transAxes, fontsize=8, color=TEXT)
    return fig


# --------------------------------------------------------------------------- #
# 7. Bootstrap intervals
# --------------------------------------------------------------------------- #


def bootstrap_intervals(summary: pd.DataFrame, ci: float = 0.90, figsize: tuple[float, float] = (7.2, 2.8)) -> Figure:
    """Point estimates with percentile bootstrap intervals, one row per measure and level."""
    apply_style()
    rows = summary.sort_values(["level", "measure"], ascending=[False, False]).reset_index(drop=True)
    labels = [f"{m} {l:.0%}" for m, l in zip(rows["measure"], rows["level"])]
    y = np.arange(len(rows))[::-1]
    colors = [RUST if m == "ES" else BLUE for m in rows["measure"]]
    fig, ax = plt.subplots(figsize=figsize)
    for yi, (_, r), c in zip(y, rows.iterrows(), colors):
        ax.hlines(yi, r["lower"], r["upper"], color=c, lw=1.6, zorder=2)
        ax.vlines([r["lower"], r["upper"]], yi - 0.12, yi + 0.12, color=c, lw=1.0, zorder=2)
        ax.scatter(r["estimate"], yi, s=36, color=c, zorder=3, lw=0)
        ax.text(r["upper"], yi, f"  {r['lower']:.1%} – {r['upper']:.1%}", va="center", fontsize=7.5, color=TEXT_MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, rows["upper"].max() * 1.45)
    ax.xaxis.set_major_formatter(_pct(0))
    ax.set_xlabel(f"One-day loss, % of value (point estimate and {ci:.0%} bootstrap interval)")
    ax.grid(False)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    return fig


# --------------------------------------------------------------------------- #
# 8. Simulation: terminal distributions (small multiples)
# --------------------------------------------------------------------------- #


def simulation_terminal(
    results: dict[str, SimulationResult],
    v0: float = cfg.INITIAL_VALUE,
    figsize: tuple[float, float] = (7.2, 4.8),
) -> Figure:
    """One panel per engine, shared x-axis, with the 1st and 5th percentiles marked."""
    apply_style()
    names = list(results)
    all_t = np.concatenate([results[n].terminal for n in names])
    lo, hi = np.quantile(all_t, [0.002, 0.995])
    bins = np.linspace(lo, hi, 80)
    fig, axes = plt.subplots(len(names), 1, figsize=figsize, sharex=True, gridspec_kw={"hspace": 0.35})
    for ax, name in zip(np.atleast_1d(axes), names):
        t = results[name].terminal
        q01, q05, q50 = np.quantile(t, [0.01, 0.05, 0.50])
        bw = bins[1] - bins[0]
        ax.hist(t, bins=bins, density=True, color=GRAY_FILL, edgecolor="white", lw=0.3)
        tail = t[t <= q05]
        # Same vertical scale as the full density: counts / (N * bin width).
        ax.hist(tail, bins=bins, weights=np.full(tail.size, 1.0 / (t.size * bw)), color=RUST_LIGHT, edgecolor="white", lw=0.3)
        ax.axvline(v0, color=GRAY, lw=0.8)
        ax.axvline(q50, color=BLUE, lw=1.2)
        ax.axvline(q01, color=RUST, lw=1.0)
        ax.axvline(q05, color=RUST, lw=1.0, alpha=0.6)
        ymax = ax.get_ylim()[1]
        ax.text(q50, ymax * 0.96, f" median {q50/v0-1:+.0%}", color=BLUE, fontsize=7.5, va="top")
        ax.text(q01, ymax * 0.96, f"1st pct {q01/v0-1:+.0%} ", color=RUST, fontsize=7.5, va="top", ha="right")
        ax.text(q05, ymax * 0.96, f" 5th pct {q05/v0-1:+.0%}", color=RUST, fontsize=7.5, va="top", ha="left", alpha=0.85)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.grid(False)
        ax.set_title(name, color=TEXT)
    ax = np.atleast_1d(axes)[-1]
    ax.xaxis.set_major_formatter(FuncFormatter(_thousands))
    ax.set_xlabel(f"Portfolio value after one year ({cfg.CURRENCY}; start {v0/1000:,.0f}k, gray line)")
    return fig


def simulation_fan(
    results: dict[str, SimulationResult],
    v0: float = cfg.INITIAL_VALUE,
    figsize: tuple[float, float] = (10, 3.4),
) -> Figure:
    """Side-by-side fan charts (1–99, 5–95, 25–75 bands and the median) for each engine."""
    apply_style()
    names = list(results)
    fig, axes = plt.subplots(1, len(names), figsize=figsize, sharey=True, gridspec_kw={"wspace": 0.10})
    for i, (ax, name) in enumerate(zip(np.atleast_1d(axes), names)):
        fan = results[name].fan((0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99))
        d = np.arange(fan.shape[0])
        ax.fill_between(d, fan[0.01], fan[0.99], color=BLUE_LIGHT, lw=0)
        ax.fill_between(d, fan[0.05], fan[0.95], color=BLUE_MID, lw=0, alpha=0.75)
        ax.fill_between(d, fan[0.25], fan[0.75], color=BLUE, lw=0, alpha=0.55)
        ax.plot(d, fan[0.50], color=INK, lw=1.0)
        ax.axhline(v0, color=GRAY, lw=0.8)
        ax.plot(d, fan[0.01], color=RUST, lw=0.9)
        ax.set_title(name, color=TEXT)
        ax.set_xlim(0, d[-1])
        ax.set_xticks([0, 63, 126, 189, 252])
        ax.set_xlabel("Trading days ahead")
        ax.grid(False)
        end = fan.iloc[-1]
        if i == len(names) - 1:
            for q, c, txt in ((0.99, BLUE, "99th pct"), (0.50, INK, "median"), (0.01, RUST, "1st pct")):
                ax.text(d[-1], end[q], f" {txt}", color=c, fontsize=7, va="center", clip_on=False)
        else:
            for q, c, txt in ((0.99, BLUE, "99th"), (0.50, INK, "median"), (0.01, RUST, "1st")):
                ax.text(d[-1] - 3, end[q], txt, color=c, fontsize=6.5, va="bottom", ha="right")
    np.atleast_1d(axes)[0].yaxis.set_major_formatter(FuncFormatter(_thousands))
    np.atleast_1d(axes)[0].set_ylabel(f"Portfolio value ({cfg.CURRENCY})")
    return fig


# --------------------------------------------------------------------------- #
# 9. Stress scenarios
# --------------------------------------------------------------------------- #


def stress_losses(
    table: pd.DataFrame,
    reference: dict[str, float] | None = None,
    value: float = cfg.INITIAL_VALUE,
    figsize: tuple[float, float] = (7.2, 3.2),
) -> Figure:
    """Horizontal bars of scenario losses in currency, sorted, with optional reference lines.

    ``reference`` maps a label (e.g. ``"1-day 99% VaR"``) to a loss fraction.
    """
    apply_style()
    t = table.sort_values("loss_value", ascending=True).reset_index(drop=True)
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(y, t["loss_value"], height=0.55, color=RUST, lw=0)
    for yi, (loss, ret) in enumerate(zip(t["loss_value"], t["portfolio_return"])):
        ax.text(loss, yi, f"  {loss/1000:,.0f}k  ({ret:.1%})", va="center", fontsize=7.5, color=TEXT)
    ax.set_yticks(y)
    ax.set_yticklabels(t["scenario"])
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_formatter(FuncFormatter(_thousands))
    ax.set_xlabel(f"Immediate loss on {value/1000:,.0f}k ({cfg.CURRENCY})")
    ax.set_xlim(0, t["loss_value"].max() * 1.32)
    ax.grid(axis="x")
    ax.grid(False, axis="y")
    if reference:
        for i, (lab, frac) in enumerate(reference.items()):
            ax.axvline(frac * value, color=INK, lw=0.8)
            ax.text(frac * value, 1.02 + 0.13 * i, f" {lab}", color=INK, fontsize=7, va="bottom",
                    transform=ax.get_xaxis_transform(), clip_on=False)
    ax.set_ylim(-0.6, len(t) - 0.2)
    return fig


# --------------------------------------------------------------------------- #
# 10. Correlation matrix
# --------------------------------------------------------------------------- #


def correlation_matrix(corr: pd.DataFrame, figsize: tuple[float, float] = (3.6, 3.2)) -> Figure:
    """Lower-triangle correlation cells with values, diverging blue (+) / rust (−)."""
    apply_style()
    c = corr.copy()
    labels = [_label(x) for x in c.columns]
    n = len(labels)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("bwr_muted", [RUST, "#f0efec", BLUE])
    fig, ax = plt.subplots(figsize=figsize)
    for i in range(n):
        for j in range(n):
            if j > i:
                continue
            v = c.iat[i, j]
            face = cmap((v + 1) / 2)
            ax.add_patch(mpl.patches.Rectangle((j, n - 1 - i), 1, 1, facecolor=face, edgecolor="white", lw=2))
            lum = 0.299 * face[0] + 0.587 * face[1] + 0.114 * face[2]
            ax.text(j + 0.5, n - 1 - i + 0.5, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if lum < 0.55 else TEXT)
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_yticklabels(labels[::-1], fontsize=7.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(False)
    ax.set_aspect("equal")
    return fig
