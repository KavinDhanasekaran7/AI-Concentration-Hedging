"""Simplified Kritzman-Turkington (2025) replication on the 49 industries.

Builds a monthly market-concentration series (effective number of industries),
regresses future market outcomes on it with Newey-West errors, and runs a simple
concentration-timing rule against buy-and-hold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src import config, metrics


def industry_weights(n_firms: pd.DataFrame, avg_size: pd.DataFrame) -> pd.DataFrame:
    """Monthly industry market-cap weights from firm counts and average firm size.

    Args:
        n_firms: number of firms per industry per month.
        avg_size: average firm size per industry per month.
    Returns:
        DataFrame of weights per industry per month, rows summing to 1.
    """
    market_cap = (n_firms * avg_size).where(lambda frame: frame > 0)
    return market_cap.div(market_cap.sum(axis=1), axis=0)


def effective_number(weights: pd.DataFrame) -> pd.Series:
    """Effective number of industries, 1 / sum of squared weights (Herfindahl).

    Args:
        weights: industry weights per month (rows sum to 1).
    Returns:
        Series of the effective number of industries per month.
    """
    return 1.0 / (weights.pow(2).sum(axis=1))


def _forward_return(monthly_excess: pd.Series, horizon: int) -> pd.Series:
    """Cumulative excess return over the next `horizon` months (aligned to now).

    Args:
        monthly_excess: monthly excess market returns (fraction).
        horizon: number of forward months.
    Returns:
        Series of forward cumulative excess returns, NaN where the window runs off.
    """
    forward = (1.0 + monthly_excess).rolling(horizon).apply(np.prod, raw=True) - 1.0
    return forward.shift(-horizon)


def _forward_volatility(monthly_return: pd.Series, horizon: int) -> pd.Series:
    """Annualised realised volatility over the next `horizon` months.

    Args:
        monthly_return: monthly market returns (fraction).
        horizon: number of forward months.
    Returns:
        Series of forward realised volatility, NaN where the window runs off.
    """
    vol = monthly_return.rolling(horizon).std(ddof=1) * np.sqrt(config.MONTHS_PER_YEAR)
    return vol.shift(-horizon)


def _forward_max_drawdown(monthly_return: pd.Series, horizon: int) -> pd.Series:
    """Max drawdown realised over the next `horizon` months (negative fraction).

    Args:
        monthly_return: monthly market returns (fraction).
        horizon: number of forward months.
    Returns:
        Series of forward max drawdowns, NaN where the window runs off.
    """
    def window_drawdown(window: np.ndarray) -> float:
        wealth = np.cumprod(1.0 + window)
        return float((wealth / np.maximum.accumulate(wealth) - 1.0).min())

    drawdown = monthly_return.rolling(horizon).apply(window_drawdown, raw=True)
    return drawdown.shift(-horizon)


def predictive_regression(concentration: pd.Series, outcome: pd.Series,
                          newey_west_lags: int) -> dict[str, float]:
    """Regress a future outcome on standardised concentration (Newey-West errors).

    Args:
        concentration: monthly effective-number series (the predictor).
        outcome: the forward outcome to predict, aligned to the same index.
        newey_west_lags: HAC lag length (set to the forecast horizon).
    Returns:
        Dict with the slope coefficient, its t-stat and the regression R-squared.
    """
    data = pd.concat([concentration, outcome], axis=1, keys=["c", "y"]).dropna()
    standardised = (data["c"] - data["c"].mean()) / data["c"].std(ddof=0)
    design = sm.add_constant(standardised.rename("concentration"))
    model = sm.OLS(data["y"], design).fit(cov_type="HAC",
                                           cov_kwds={"maxlags": newey_west_lags})
    return {
        "coef": float(model.params["concentration"]),
        "t_stat": float(model.tvalues["concentration"]),
        "r_squared": float(model.rsquared),
        "n_obs": int(model.nobs),
    }


def run_predictive_table(concentration: pd.Series, factors: pd.DataFrame) -> pd.DataFrame:
    """Build the full table of concentration-predictability regressions.

    Args:
        concentration: monthly effective-number series.
        factors: Fama-French monthly factor table (percent units).
    Returns:
        DataFrame indexed by outcome with coef, t_stat, r_squared and n_obs,
        testing whether concentration predicts future returns, vol or drawdown.
    """
    excess = factors["Mkt-RF"] / 100.0
    total = factors["Mkt"] / 100.0
    outcomes = {
        "excess_return_1m": (_forward_return(excess, 1), 1),
        "excess_return_3m": (_forward_return(excess, 3), 3),
        "excess_return_12m": (_forward_return(excess, 12), 12),
        "realised_vol_12m": (_forward_volatility(total, 12), 12),
        "max_drawdown_12m": (_forward_max_drawdown(total, 12), 12),
    }
    rows = {name: predictive_regression(concentration, series, lags)
            for name, (series, lags) in outcomes.items()}
    return pd.DataFrame(rows).T


def concentration_timing_backtest(concentration: pd.Series,
                                  factors: pd.DataFrame) -> pd.DataFrame:
    """Compare a concentration-timing rule with buy-and-hold equity.

    The rule holds 50% equity / 50% T-bills whenever concentration sits in its
    lowest expanding-window quintile (most concentrated, past data only), else 100%
    equity. Returns are monthly and the comparison uses only information available
    at each month-end.

    Args:
        concentration: monthly effective-number series.
        factors: Fama-French monthly factor table (percent units).
    Returns:
        DataFrame indexed by strategy ('timing', 'buy_and_hold') with CAGR, vol,
        Sharpe and max drawdown (monthly metrics).
    """
    equity = (factors["Mkt"] / 100.0).reindex(concentration.index).dropna()
    bills = (factors["RF"] / 100.0).reindex(equity.index)
    concentration = concentration.reindex(equity.index)

    low_quintile = concentration.expanding(min_periods=24).quantile(0.20)
    is_concentrated = (concentration <= low_quintile).shift(1, fill_value=False)
    timing = np.where(is_concentrated, 0.5 * equity + 0.5 * bills, equity)
    timing = pd.Series(timing, index=equity.index)

    rows = {
        "timing": metrics.summary_metrics(timing, config.MONTHS_PER_YEAR, bills),
        "buy_and_hold": metrics.summary_metrics(equity, config.MONTHS_PER_YEAR, bills),
    }
    columns = ["ann_return", "ann_vol", "sharpe", "max_drawdown"]
    return pd.DataFrame(rows).T[columns]
