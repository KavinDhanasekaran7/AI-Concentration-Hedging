"""Tail-structure measures for the dial portfolios (hypothesis H1).

Two measures: how much of the portfolio's worst-day loss comes from the top sleeve,
and the effective number of independent bets (Meucci-style) in calm versus stress.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def worst_day_mask(portfolio_returns: pd.Series, level: float) -> pd.Series:
    """Boolean mask of the portfolio's worst (1 - level) fraction of days.

    Args:
        portfolio_returns: daily portfolio returns.
        level: confidence level, e.g. 0.95 selects the worst 5% of days.
    Returns:
        Boolean Series, True on the worst-tail days.
    """
    threshold = portfolio_returns.quantile(1.0 - level)
    return portfolio_returns <= threshold


def top_sleeve_es_share(portfolio_returns: pd.Series, sleeve_contribution: pd.Series,
                        level: float) -> float:
    """Share of the portfolio's tail loss contributed by the top sleeve.

    Mean top-sleeve contribution on the worst-tail days divided by the mean portfolio
    return on those days. Values above the sleeve's weight mean the tail is sleeve-driven.

    Args:
        portfolio_returns: daily portfolio returns.
        sleeve_contribution: daily portfolio return attributable to the top sleeve.
        level: confidence level, e.g. 0.95.
    Returns:
        Fraction of the tail loss coming from the top sleeve.
    """
    mask = worst_day_mask(portfolio_returns, level)
    tail_portfolio = portfolio_returns[mask].mean()
    tail_sleeve = sleeve_contribution.reindex(portfolio_returns.index)[mask].mean()
    return float(tail_sleeve / tail_portfolio) if tail_portfolio else float("nan")


def effective_number_of_bets(contributions: pd.DataFrame) -> float:
    """Meucci-style effective number of bets from per-asset contribution returns.

    Eigen-decompose the contribution covariance; with p_k the normalised eigenvalues,
    ENB = exp(-Σ p_k ln p_k). It falls toward 1 when risk collapses onto one factor.

    Args:
        contributions: daily per-asset contribution returns (zero columns ignored).
    Returns:
        The effective number of bets over the sample (NaN if fewer than 2 assets).
    """
    active = contributions.loc[:, contributions.abs().sum() > 0].dropna()
    if active.shape[1] < 2 or active.shape[0] < active.shape[1]:
        return float("nan")
    eigenvalues = np.linalg.eigvalsh(np.cov(active.to_numpy(), rowvar=False))
    eigenvalues = eigenvalues[eigenvalues > 0]
    proportions = eigenvalues / eigenvalues.sum()
    entropy = -np.sum(proportions * np.log(proportions))
    return float(np.exp(entropy))


def enb_calm_vs_stress(contributions: pd.DataFrame,
                       stress_episodes: list[tuple[str, str, str]] | None = None) -> dict:
    """Effective number of bets over the calm full sample versus each stress episode.

    Args:
        contributions: daily per-asset contribution returns.
        stress_episodes: list of (label, start, end); defaults to config.STRESS_EPISODES.
    Returns:
        Dict mapping 'full_sample' and each stress label to its effective number of bets.
    """
    episodes = stress_episodes if stress_episodes is not None else config.STRESS_EPISODES
    result = {"full_sample": effective_number_of_bets(contributions)}
    for label, start, end in episodes:
        window = contributions.loc[start:end]
        if len(window) >= 10:
            result[label] = effective_number_of_bets(window)
    return result


def sleeve_contribution(contributions: pd.DataFrame,
                        rest_ticker: str = config.REST_OF_MARKET_TICKER) -> pd.Series:
    """Daily portfolio return attributable to the top sleeve (all names but RSP).

    Args:
        contributions: daily per-asset contribution returns.
        rest_ticker: the rest-of-market column to exclude.
    Returns:
        Series of daily top-sleeve contribution to the portfolio return.
    """
    sleeve_columns = [c for c in contributions.columns if c != rest_ticker]
    return contributions[sleeve_columns].sum(axis=1).rename("sleeve_contribution")
