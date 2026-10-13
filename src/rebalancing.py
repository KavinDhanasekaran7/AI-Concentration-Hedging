"""Rules for managing concentration within the dial portfolios (section 3.3).

Three rules, each producing a monthly target-weight matrix to feed the backtest:
a 5% single-name cap, equal weighting inside the top sleeve, and a clustered
inverse-volatility allocation over the eleven assets (ten stocks plus RSP).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from src import config


def cap_single_names(dial_weights: pd.DataFrame, cap: float = config.SINGLE_NAME_CAP,
                     rest_ticker: str = config.REST_OF_MARKET_TICKER) -> pd.DataFrame:
    """Cap each single stock at a fraction of the portfolio, excess to rest-of-market.

    Args:
        dial_weights: month-end dial target weights (sleeve stocks plus RSP).
        cap: maximum weight for any single stock.
        rest_ticker: the rest-of-market column that absorbs the excess.
    Returns:
        Capped month-end target weights, rows still summing to 1.
    """
    capped = dial_weights.copy()
    stock_columns = [c for c in capped.columns if c != rest_ticker]
    excess = (capped[stock_columns] - cap).clip(lower=0).sum(axis=1)
    capped[stock_columns] = capped[stock_columns].clip(upper=cap)
    capped[rest_ticker] = capped[rest_ticker] + excess
    return capped


def equal_weight_sleeve(sleeve_weights: pd.DataFrame, w_top: float,
                        rest_ticker: str = config.REST_OF_MARKET_TICKER) -> pd.DataFrame:
    """Weight the top sleeve equally (sleeve total unchanged at w_top), rest gets 1-w_top.

    Args:
        sleeve_weights: within-sleeve month-end weights (non-zero where held).
        w_top: fraction allocated to the top sleeve (the dial level).
        rest_ticker: the rest-of-market column.
    Returns:
        Month-end target weights with an equal-weighted sleeve, rows summing to 1.
    """
    weights = sleeve_weights * 0.0
    for date, row in sleeve_weights.iterrows():
        held = row[row > 0].index
        if len(held):
            weights.loc[date, held] = w_top / len(held)
    if rest_ticker not in weights.columns:
        weights[rest_ticker] = 0.0
    weights[rest_ticker] = 1.0 - w_top
    return weights


def _cluster_labels(correlation: pd.DataFrame, n_clusters: int) -> np.ndarray:
    """Hierarchical (average-linkage) cluster labels from a correlation matrix.

    Args:
        correlation: asset correlation matrix.
        n_clusters: number of clusters to cut to.
    Returns:
        Integer cluster label per asset (order matches the matrix).
    """
    distance = np.sqrt(0.5 * (1.0 - correlation.to_numpy()))
    np.fill_diagonal(distance, 0.0)
    linkage_matrix = linkage(squareform(distance, checks=False), method="average")
    return fcluster(linkage_matrix, t=min(n_clusters, len(correlation)), criterion="maxclust")


def _clustered_inverse_vol(volatility: pd.Series, labels: np.ndarray) -> pd.Series:
    """Inverse-vol weights within clusters, equal total weight across clusters.

    Args:
        volatility: per-asset volatility (index aligned to the labels order).
        labels: cluster label per asset.
    Returns:
        Weights summing to 1 across all assets.
    """
    weights = pd.Series(0.0, index=volatility.index)
    unique = np.unique(labels)
    for cluster in unique:
        members = volatility.index[labels == cluster]
        inverse = 1.0 / volatility[members]
        weights[members] = (inverse / inverse.sum()) / len(unique)
    return weights


def clustered_inverse_vol_weights(returns: pd.DataFrame, sleeve_by_year: dict[int, list[str]],
                                  rebalance_index: pd.DatetimeIndex,
                                  rest_ticker: str = config.REST_OF_MARKET_TICKER,
                                  n_clusters: int = config.N_CLUSTERS,
                                  window: int = config.CORRELATION_WINDOW_DAYS) -> pd.DataFrame:
    """Clustered inverse-vol target weights over each year's sleeve plus RSP.

    At every rebalance the eleven assets are clustered on their trailing-year
    correlations; weights are inverse-vol within a cluster and equal across clusters.

    Args:
        returns: daily simple returns for the candidate pool and RSP.
        sleeve_by_year: that year's sleeve membership.
        rebalance_index: month-end rebalance dates.
        rest_ticker: the rest-of-market column always included in the universe.
        n_clusters: number of clusters to cut to.
        window: trailing window (days) for correlations and volatilities.
    Returns:
        Month-end target weights over all assets, rows summing to 1.
    """
    weights = pd.DataFrame(0.0, index=rebalance_index, columns=returns.columns)
    for date in rebalance_index:
        universe = [t for t in sleeve_by_year.get(date.year, []) if t in returns.columns]
        universe = universe + [rest_ticker]
        history = returns.loc[:date, universe].tail(window).dropna(axis=1, how="any")
        if history.shape[1] < 2 or len(history) < window // 2:
            continue
        labels = _cluster_labels(history.corr(), n_clusters)
        allocation = _clustered_inverse_vol(history.std(ddof=1), labels)
        weights.loc[date, allocation.index] = allocation.values
    return weights.loc[weights.sum(axis=1) > 0]


def annual_turnover(target_weights: pd.DataFrame, returns: pd.DataFrame) -> float:
    """One-way annual turnover implied by monthly rebalancing of target weights.

    Weights drift with returns within each month; turnover at a rebalance is half the
    sum of absolute changes from the drifted weights to the new target.

    Args:
        target_weights: month-end target weights, rows summing to 1.
        returns: daily simple returns for every asset in target_weights.
    Returns:
        Average one-way turnover per year.
    """
    assets = list(target_weights.columns)
    returns = returns.reindex(columns=assets).fillna(0.0)
    rebalance_dates = list(target_weights.index)
    drifted, trades = None, []
    for i, start in enumerate(rebalance_dates):
        target = target_weights.loc[start].to_numpy()
        if drifted is not None:
            trades.append(0.5 * np.abs(target - drifted).sum())
        stop = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else returns.index.max()
        window = returns.loc[(returns.index > start) & (returns.index <= stop)]
        if window.empty:
            drifted = target
            continue
        grown = target * (1.0 + window).prod().to_numpy()
        drifted = grown / grown.sum()
    years = (rebalance_dates[-1] - rebalance_dates[0]).days / 365.25
    return float(np.sum(trades) / years) if years > 0 else float("nan")
