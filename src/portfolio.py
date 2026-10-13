"""Top-10 ranking, concentration-dial portfolios and the backtest engine.

Each year-end the candidate pool is ranked by market cap; the top names form that
year's "top sleeve". A dial portfolio blends the cap-weighted sleeve with the
equal-weight rest-of-market sleeve (RSP) at a chosen weight, rebalanced monthly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns from a price panel.

    Args:
        prices: daily prices, one column per asset.
    Returns:
        Daily simple returns with the first row dropped.
    """
    return prices.pct_change().iloc[1:]


def month_end_market_caps(prices: pd.DataFrame, shares: pd.DataFrame) -> pd.DataFrame:
    """Month-end market caps = price times latest reported shares on or before the date.

    Args:
        prices: daily prices for the candidate pool.
        shares: EDGAR shares outstanding reported on filing dates.
        shares and prices share their ticker columns.
    Returns:
        DataFrame of month-end market caps, one column per ticker.
    """
    daily_shares = shares.reindex(prices.index.union(shares.index)).ffill()
    daily_shares = daily_shares.reindex(prices.index)
    caps = prices * daily_shares[prices.columns]
    return caps.resample("ME").last()


def annual_top_sleeve(month_end_caps: pd.DataFrame, size: int) -> dict[int, list[str]]:
    """Rank by December market cap to pick each following year's top sleeve.

    Args:
        month_end_caps: month-end market caps per ticker.
        size: number of names in the sleeve (top 10).
    Returns:
        Dict mapping calendar year to that year's sleeve (ranked at the prior year-end).
    """
    december = month_end_caps[month_end_caps.index.month == 12]
    sleeve_by_year: dict[int, list[str]] = {}
    for date, caps in december.iterrows():
        ranked = caps.dropna().sort_values(ascending=False)
        sleeve_by_year[date.year + 1] = list(ranked.head(size).index)
    return sleeve_by_year


def monthly_sleeve_weights(month_end_caps: pd.DataFrame,
                           sleeve_by_year: dict[int, list[str]]) -> pd.DataFrame:
    """Within-sleeve cap weights at each month-end (sum to 1 across the sleeve).

    Args:
        month_end_caps: month-end market caps per ticker.
        sleeve_by_year: that year's sleeve membership.
    Returns:
        DataFrame indexed by month-end, one column per ticker, weights summing to 1
        across the active sleeve and zero for names not in it.
    """
    weights = pd.DataFrame(0.0, index=month_end_caps.index, columns=month_end_caps.columns)
    for date, caps in month_end_caps.iterrows():
        sleeve = sleeve_by_year.get(date.year, [])
        held = [t for t in sleeve if t in caps.index and caps[t] > 0]
        if not held:
            continue
        weights.loc[date, held] = (caps[held] / caps[held].sum()).values
    return weights.loc[weights.sum(axis=1) > 0]


def dial_weight_matrix(sleeve_weights: pd.DataFrame, w_top: float,
                       rest_ticker: str = config.REST_OF_MARKET_TICKER) -> pd.DataFrame:
    """Blend the cap-weighted sleeve with the rest-of-market sleeve at weight w_top.

    Args:
        sleeve_weights: within-sleeve month-end weights (sum to 1).
        w_top: fraction allocated to the top sleeve (the dial level).
        rest_ticker: ticker for the rest-of-market sleeve (RSP).
    Returns:
        Month-end target weights including the rest-of-market column, rows summing to 1.
    """
    weights = sleeve_weights * w_top
    if rest_ticker not in weights.columns:
        weights[rest_ticker] = 0.0
    weights[rest_ticker] = 1.0 - w_top
    return weights


def run_backtest(returns: pd.DataFrame, target_weights: pd.DataFrame) -> dict[str, object]:
    """Backtest monthly-rebalanced target weights, buy-and-hold within each month.

    Weights set on a rebalance date apply to the following days until the next
    rebalance (no look-ahead). Within a month, holdings drift with prices.

    Args:
        returns: daily simple returns for every asset that appears in target_weights.
        target_weights: month-end target weights, rows summing to 1.
    Returns:
        Dict with 'returns' (daily portfolio returns) and 'contributions'
        (daily per-asset contribution to the portfolio return).
    """
    assets = list(target_weights.columns)
    returns = returns.reindex(columns=assets).fillna(0.0)
    rebal_dates = list(target_weights.index)
    port_returns, contributions = [], []
    for i, start in enumerate(rebal_dates):
        stop = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else returns.index.max()
        window = returns.loc[(returns.index > start) & (returns.index <= stop)]
        if window.empty:
            continue
        weights = target_weights.loc[start].to_numpy()
        growth = (1.0 + window).cumprod()
        prior_growth = growth.shift(1).fillna(1.0)
        effective = prior_growth.to_numpy() * weights
        value = effective.sum(axis=1)
        contribution = effective * window.to_numpy() / value[:, None]
        contributions.append(pd.DataFrame(contribution, index=window.index, columns=assets))
        port_returns.append(pd.Series((contribution).sum(axis=1), index=window.index))
    return {
        "returns": pd.concat(port_returns).rename("portfolio"),
        "contributions": pd.concat(contributions),
    }


def build_dial_returns(returns: pd.DataFrame, sleeve_weights: pd.DataFrame,
                       start: str = config.STOCK_ANALYSIS_START) -> dict:
    """Run every dial level plus the standalone top sleeve from one weight set.

    Args:
        returns: daily simple returns for the candidate pool and RSP.
        sleeve_weights: within-sleeve month-end weights (sum to 1).
        start: analysis start date (returns before this are discarded).
    Returns:
        Dict with 'sleeve' (backtest dict for the pure top sleeve) and 'dials'
        (dict keyed by dial level -> backtest dict), all trimmed to start onward.
    """
    sleeve = run_backtest(returns, sleeve_weights)
    dials = {}
    for w_top in config.DIAL_LEVELS:
        dials[w_top] = run_backtest(returns, dial_weight_matrix(sleeve_weights, w_top))
    trim = lambda bt: {k: v.loc[v.index >= start] for k, v in bt.items()}
    return {"sleeve": trim(sleeve), "dials": {w: trim(bt) for w, bt in dials.items()}}
