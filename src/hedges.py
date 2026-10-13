"""Overlay hedges applied to the dial portfolios, with trading and financing costs.

Every overlay returns a daily P&L series (fraction of portfolio value) to add to the
dial portfolio's return. Shorts earn the T-bill rate minus a borrow spread; changing a
position costs a per-unit-turnover fee. All signals use prior-day information only.
"""
from __future__ import annotations

import pandas as pd

from src import config


def _short_overlay_pnl(short_notional: pd.Series, asset_returns: pd.Series,
                       rate_daily: pd.Series) -> pd.Series:
    """Daily P&L of a short position of given notional, net of financing and turnover.

    Args:
        short_notional: daily short size as a fraction of portfolio (>= 0, prior-day).
        asset_returns: daily returns of the shorted asset.
        rate_daily: daily simple risk-free rate.
    Returns:
        Daily overlay P&L as a fraction of portfolio value.
    """
    position = short_notional.reindex(asset_returns.index).fillna(0.0)
    held = position.shift(1).fillna(0.0)
    borrow_daily = config.SHORT_BORROW_SPREAD / config.TRADING_DAYS_PER_YEAR
    price_pnl = -held * asset_returns
    financing = held * (rate_daily.reindex(asset_returns.index).fillna(0.0) - borrow_daily)
    turnover_cost = config.OVERLAY_TURNOVER_COST * (position - position.shift(1).fillna(0.0)).abs()
    return (price_pnl + financing - turnover_cost).rename("overlay")


def trend_overlay(spy: pd.Series, spy_returns: pd.Series, rate_daily: pd.Series,
                  hedge_notional: float = config.HEDGE_NOTIONAL,
                  ma_days: int = config.TREND_MA_DAYS) -> pd.Series:
    """Short SPY at the hedge notional whenever SPY closed below its 200-day average.

    Args:
        spy: daily SPY level.
        spy_returns: daily SPY returns.
        rate_daily: daily simple risk-free rate.
        hedge_notional: short size as a fraction of portfolio when triggered.
        ma_days: moving-average window for the trend signal.
    Returns:
        Daily overlay P&L as a fraction of portfolio value.
    """
    below = (spy < spy.rolling(ma_days).mean()).shift(1).fillna(False)
    short_notional = below.astype(float) * hedge_notional
    return _short_overlay_pnl(short_notional, spy_returns, rate_daily).rename("trend")


def naive_futures_overlay(spy_returns: pd.Series, rate_daily: pd.Series,
                          hedge_notional: float = config.HEDGE_NOTIONAL) -> pd.Series:
    """Permanently short SPY at the hedge notional (never re-estimated).

    Args:
        spy_returns: daily SPY returns.
        rate_daily: daily simple risk-free rate.
        hedge_notional: constant short size as a fraction of portfolio.
    Returns:
        Daily overlay P&L as a fraction of portfolio value.
    """
    short_notional = pd.Series(hedge_notional, index=spy_returns.index)
    return _short_overlay_pnl(short_notional, spy_returns, rate_daily).rename("naive")


def rolling_beta(asset_returns: pd.Series, factor_returns: pd.Series,
                 window: int = config.BETA_WINDOW_DAYS) -> pd.Series:
    """Trailing beta of an asset on a factor over a rolling window.

    Args:
        asset_returns: daily returns of the asset (e.g. the top sleeve).
        factor_returns: daily returns of the factor (e.g. QQQ).
        window: trailing window in days.
    Returns:
        Daily rolling beta series.
    """
    covariance = asset_returns.rolling(window).cov(factor_returns)
    variance = factor_returns.rolling(window).var()
    return covariance / variance


def concentration_overlay(w_top: float, sleeve_returns: pd.Series, qqq_returns: pd.Series,
                          rate_daily: pd.Series, floor: float = config.CONCENTRATION_HEDGE_FLOOR,
                          window: int = config.BETA_WINDOW_DAYS) -> pd.Series:
    """Short QQQ scaled by excess concentration and the sleeve's beta to QQQ.

    Notional = max(0, w_top - floor) x beta(sleeve on QQQ). The hedge grows with both
    the concentration above the floor and how tech-like the sleeve has become.

    Args:
        w_top: the dial level (top-sleeve weight).
        sleeve_returns: daily top-sleeve returns.
        qqq_returns: daily QQQ returns.
        rate_daily: daily simple risk-free rate.
        floor: concentration level below which no hedge is applied.
        window: trailing window for the beta estimate.
    Returns:
        Daily overlay P&L as a fraction of portfolio value.
    """
    beta = rolling_beta(sleeve_returns, qqq_returns, window).shift(1).clip(lower=0.0)
    short_notional = max(0.0, w_top - floor) * beta
    return _short_overlay_pnl(short_notional.fillna(0.0), qqq_returns, rate_daily).rename("concentration")


def concentration_overlay_variant(w_top: float, sleeve_returns: pd.Series,
                                  rsp_returns: pd.Series, spy_returns: pd.Series,
                                  qqq_returns: pd.Series,
                                  floor: float = config.CONCENTRATION_HEDGE_FLOOR,
                                  window: int = config.BETA_WINDOW_DAYS) -> pd.Series:
    """Concentration hedge variant: long RSP / short SPY at the same notional.

    Args:
        w_top: the dial level (top-sleeve weight).
        sleeve_returns: daily top-sleeve returns.
        rsp_returns: daily RSP returns (long leg).
        spy_returns: daily SPY returns (short leg).
        qqq_returns: daily QQQ returns (for the sizing beta).
        floor: concentration level below which no hedge is applied.
        window: trailing window for the beta estimate.
    Returns:
        Daily overlay P&L as a fraction of portfolio value.
    """
    beta = rolling_beta(sleeve_returns, qqq_returns, window).shift(1).clip(lower=0.0).fillna(0.0)
    notional = max(0.0, w_top - floor) * beta
    held = notional.shift(1).fillna(0.0)
    spread = (rsp_returns - spy_returns).reindex(held.index).fillna(0.0)
    turnover_cost = config.OVERLAY_TURNOVER_COST * (notional - notional.shift(1).fillna(0.0)).abs()
    return (held * spread - turnover_cost).rename("concentration_variant")
