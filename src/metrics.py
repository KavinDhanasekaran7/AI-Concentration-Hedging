"""Performance and risk metrics on return series.

All functions take simple (not log) returns unless stated, and a periods-per-year
argument so the same code serves daily, weekly or monthly series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def cumulative_wealth(returns: pd.Series) -> pd.Series:
    """Growth of 1 unit from a simple-return series.

    Args:
        returns: periodic simple returns.
    Returns:
        Cumulative wealth index starting just after the first return.
    """
    return (1.0 + returns.fillna(0.0)).cumprod()


def cagr(returns: pd.Series, periods_per_year: int) -> float:
    """Compound annual growth rate of a simple-return series.

    Args:
        returns: periodic simple returns.
        periods_per_year: 252 daily, 52 weekly, 12 monthly.
    Returns:
        Annualised compound growth rate.
    """
    clean = returns.dropna()
    if clean.empty:
        return float("nan")
    wealth = float((1.0 + clean).prod())
    years = len(clean) / periods_per_year
    return wealth ** (1.0 / years) - 1.0 if years > 0 else float("nan")


def annualised_volatility(returns: pd.Series, periods_per_year: int) -> float:
    """Annualised standard deviation of a simple-return series.

    Args:
        returns: periodic simple returns.
        periods_per_year: 252 daily, 52 weekly, 12 monthly.
    Returns:
        Annualised volatility.
    """
    return float(returns.dropna().std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, periods_per_year: int,
                 risk_free: pd.Series | float = 0.0) -> float:
    """Annualised Sharpe ratio using periodic excess returns.

    Args:
        returns: periodic simple returns.
        periods_per_year: 252 daily, 52 weekly, 12 monthly.
        risk_free: matching-frequency risk-free rate, aligned series or scalar.
    Returns:
        Annualised Sharpe ratio.
    """
    excess = returns.dropna()
    rf = risk_free.reindex(excess.index) if isinstance(risk_free, pd.Series) else risk_free
    excess = excess - rf
    std = excess.std(ddof=1)
    if not std:
        return float("nan")
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough decline of the cumulative wealth index.

    Args:
        returns: periodic simple returns.
    Returns:
        Maximum drawdown as a negative fraction (e.g. -0.55 for a 55% fall).
    """
    wealth = cumulative_wealth(returns)
    running_peak = wealth.cummax()
    return float((wealth / running_peak - 1.0).min())


def value_at_risk(returns: pd.Series, level: float) -> float:
    """Historical Value-at-Risk at a confidence level (loss as a negative number).

    Args:
        returns: periodic simple returns.
        level: confidence level, e.g. 0.95.
    Returns:
        The (1 - level) empirical quantile of returns.
    """
    return float(returns.dropna().quantile(1.0 - level))


def expected_shortfall(returns: pd.Series, level: float) -> float:
    """Historical expected shortfall (mean loss beyond VaR), a negative number.

    Args:
        returns: periodic simple returns.
        level: confidence level, e.g. 0.95.
    Returns:
        Mean of returns at or below the VaR quantile.
    """
    clean = returns.dropna()
    threshold = clean.quantile(1.0 - level)
    tail = clean[clean <= threshold]
    return float(tail.mean()) if not tail.empty else float("nan")


def summary_metrics(returns: pd.Series, periods_per_year: int,
                    risk_free: pd.Series | float = 0.0) -> dict[str, float]:
    """Bundle the headline performance and tail metrics for a return series.

    Args:
        returns: periodic simple returns.
        periods_per_year: 252 daily, 52 weekly, 12 monthly.
        risk_free: matching-frequency risk-free rate, aligned series or scalar.
    Returns:
        Dict of annualised return, vol, Sharpe, max drawdown, VaR/ES and worst period.
    """
    clean = returns.dropna()
    metrics = {
        "ann_return": cagr(clean, periods_per_year),
        "ann_vol": annualised_volatility(clean, periods_per_year),
        "sharpe": sharpe_ratio(clean, periods_per_year, risk_free),
        "max_drawdown": max_drawdown(clean),
        "worst_period": float(clean.min()) if not clean.empty else float("nan"),
    }
    for level in config.VAR_LEVELS:
        tag = f"{level * 100:g}"            # 0.95 -> "95", 0.975 -> "97.5"
        metrics[f"VaR{tag}"] = value_at_risk(clean, level)
        metrics[f"ES{tag}"] = expected_shortfall(clean, level)
    return metrics
