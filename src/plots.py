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
