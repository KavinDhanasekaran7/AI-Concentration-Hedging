"""All project figures: one function per figure, consistent colours, saved to disk.

Each function builds a single clear message, labels its axes with units, titles the
chart with the finding, saves a 150-dpi PNG to results/figures/ and returns the Axes.
Colours are defined once here so every figure reads as one system.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from src import config

# One palette for the whole project (colour-blind-safe, defined once).
COLORS: dict[str, str] = {
    "primary": "#1f5fa8",     # deep blue: main series
    "accent": "#d1495b",      # red: stress / tail / the concentrated case
    "neutral": "#6c757d",     # grey: benchmark / do-nothing
    "highlight": "#e6a817",   # amber: the "today's market" marker
    "positive": "#2a9d8f",    # teal: protection / diversified case
    "muted": "#adb5bd",       # light grey: gridlines, context
}
DIAL_SEQUENCE: list[str] = ["#9ecae1", "#4292c6", "#d1495b", "#7a0828"]  # 0.20->0.50


def _save(fig: plt.Figure, filename: str) -> None:
    """Save a figure as a 150-dpi PNG under results/figures/.

    Args:
        fig: the figure to save.
        filename: output filename (PNG).
    """
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / filename, dpi=150, bbox_inches="tight")


def plot_concentration_over_time(concentration: pd.Series,
                                 filename: str = "concentration_over_time.png") -> plt.Axes:
    """Plot the effective number of industries over the full sample.

    Args:
        concentration: monthly effective-number series.
        filename: output PNG name.
    Returns:
        The Matplotlib Axes.
    """
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(concentration.index, concentration.values, color=COLORS["primary"], lw=1.3)
    ax.fill_between(concentration.index, concentration.values, concentration.min(),
                    color=COLORS["primary"], alpha=0.08)
    ax.set_title("US market concentration: fewer effective industries means a more "
                 "concentrated market", fontsize=11)
    ax.set_xlabel("Year")
    ax.set_ylabel("Effective number of industries (1 / Σ wᵢ²)")
    ax.grid(True, color=COLORS["muted"], alpha=0.4, lw=0.5)
    _save(fig, filename)
    return ax


def plot_tail_share_vs_concentration(share_by_dial: pd.Series,
                                     filename: str = "tail_share_vs_concentration.png") -> plt.Axes:
    """Plot the top sleeve's share of tail (ES95) loss against the dial level.

    Args:
        share_by_dial: ES95 tail-share indexed by dial level (fraction).
        filename: output PNG name.
    Returns:
        The Matplotlib Axes.
    """
    levels = share_by_dial.index.to_numpy()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(levels, levels, color=COLORS["neutral"], ls="--", lw=1,
            label="If tail loss matched sleeve weight")
    ax.plot(levels, share_by_dial.to_numpy(), color=COLORS["accent"], marker="o", lw=2,
            label="Actual sleeve share of ES95 loss")
    ax.axvline(config.DIAL_MARKET_LEVEL, color=COLORS["highlight"], lw=1.2, alpha=0.9)
    ax.text(config.DIAL_MARKET_LEVEL, ax.get_ylim()[0], "  ≈ today's market",
            color=COLORS["highlight"], va="bottom", fontsize=9)
    ax.set_title("More concentration, more of the tail loss comes from the top names",
                 fontsize=11)
    ax.set_xlabel("Top-sleeve weight (concentration dial)")
    ax.set_ylabel("Top-sleeve share of worst-5%-day loss (ES95)")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.grid(True, color=COLORS["muted"], alpha=0.4, lw=0.5)
    ax.legend(frameon=False, fontsize=9)
    _save(fig, filename)
    return ax


def plot_enb_calm_vs_stress(enb_table: pd.DataFrame,
                            filename: str = "enb_calm_vs_stress.png") -> plt.Axes:
    """Plot the effective number of bets, full sample versus stress episodes.

    Args:
        enb_table: DataFrame indexed by regime with one column per dial level.
        filename: output PNG name.
    Returns:
        The Matplotlib Axes.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = dict(zip(enb_table.columns, DIAL_SEQUENCE))
    x = range(len(enb_table.index))
    width = 0.8 / len(enb_table.columns)
    for j, level in enumerate(enb_table.columns):
        offsets = [i + j * width for i in x]
        ax.bar(offsets, enb_table[level].to_numpy(), width=width,
               color=colors[level], label=f"dial {level:.0%}")
    ax.set_title("Effective number of bets collapses toward one in stress", fontsize=11)
    ax.set_ylabel("Effective number of bets (Meucci)")
    ax.set_xticks([i + 0.4 - width / 2 for i in x])
    ax.set_xticklabels(enb_table.index, rotation=30, ha="right", fontsize=8)
    ax.axhline(1.0, color=COLORS["neutral"], lw=0.8, ls=":")
    ax.grid(True, axis="y", color=COLORS["muted"], alpha=0.4, lw=0.5)
    ax.legend(frameon=False, fontsize=8, ncol=4)
    _save(fig, filename)
    return ax


def plot_pput_calibration(modelled: pd.Series, benchmark: pd.Series,
                          filename: str = "pput_calibration.png") -> plt.Axes:
    """Plot the modelled index+put series against the Cboe PPUT benchmark (growth of 1).

    Args:
        modelled: modelled monthly returns of index + calibrated monthly put.
        benchmark: PPUT monthly returns.
        filename: output PNG name.
    Returns:
        The Matplotlib Axes.
    """
    aligned = pd.concat([modelled.rename("model"), benchmark.rename("pput")], axis=1).dropna()
    wealth = (1.0 + aligned).cumprod()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(wealth.index, wealth["model"], color=COLORS["primary"], lw=1.6,
            label="Modelled: index + calibrated monthly 5% put")
    ax.plot(wealth.index, wealth["pput"], color=COLORS["accent"], lw=1.3, ls="--",
            label="Cboe PPUT benchmark")
    ax.set_title("Calibrated put model tracks the Cboe PPUT benchmark", fontsize=11)
    ax.set_xlabel("Year")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, color=COLORS["muted"], alpha=0.4, lw=0.5)
    ax.legend(frameon=False, fontsize=9)
    _save(fig, filename)
    return ax


def plot_cost_vs_protection(scorecard: pd.DataFrame,
                            filename: str = "cost_vs_protection.png") -> plt.Axes:
    """Scatter the tail-risk removed against the carry cost for every dial x strategy.

    Args:
        scorecard: a (dial x strategy) scorecard with carry_cost and ES95_reduction_pp.
        filename: output PNG name.
    Returns:
        The Matplotlib Axes.
    """
    strategy_colors = {
        "put": COLORS["primary"], "trend": COLORS["positive"],
        "concentration": COLORS["accent"], "naive": COLORS["neutral"],
        "put_ladder": COLORS["highlight"], "cvar_mix": "#5a3e8e",
    }
    fig, ax = plt.subplots(figsize=(8, 6))
    for (dial, strategy), row in scorecard.iterrows():
        color = strategy_colors.get(strategy, COLORS["muted"])
        ax.scatter(row["carry_cost"] * 100, row["ES95_reduction_pp"], color=color, s=55,
                   alpha=0.85)
        ax.annotate(dial, (row["carry_cost"] * 100, row["ES95_reduction_pp"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=c, label=s)
               for s, c in strategy_colors.items() if s in scorecard.index.get_level_values(1)]
    ax.legend(handles=handles, frameon=False, fontsize=9, title="strategy")
    ax.set_title("Tail risk removed versus the carry paid for it (up-left is better)",
                 fontsize=11)
    ax.set_xlabel("Annual carry cost (% of portfolio, calm months)")
    ax.set_ylabel("ES95 reduction (percentage points)")
    ax.grid(True, color=COLORS["muted"], alpha=0.4, lw=0.5)
    _save(fig, filename)
    return ax
