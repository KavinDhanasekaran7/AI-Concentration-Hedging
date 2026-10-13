"""Black-Scholes option pricing and the put-protection overlays.

Puts on the S&P 500 are priced with a VIX-based volatility (sigma = m x VIX/100),
rolled at expiry and marked to market daily. The scalar m is calibrated so a modelled
"index + monthly 5% put" series tracks the Cboe PPUT benchmark.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from src import config


def black_scholes_put(spot: float, strike: float, sigma: float, maturity: float,
                      rate: float) -> float:
    """Black-Scholes European put price (no dividends).

    Args:
        spot: underlying price.
        strike: strike price.
        sigma: annualised volatility.
        maturity: time to expiry in years.
        rate: annual continuously-compounded risk-free rate.
    Returns:
        The put price; its intrinsic value if maturity or sigma is non-positive.
    """
    if maturity <= 0 or sigma <= 0:
        return max(strike - spot, 0.0)
    d1 = (np.log(spot / strike) + (rate + 0.5 * sigma ** 2) * maturity) / (sigma * np.sqrt(maturity))
    d2 = d1 - sigma * np.sqrt(maturity)
    return float(strike * np.exp(-rate * maturity) * norm.cdf(-d2) - spot * norm.cdf(-d1))


def black_scholes_call(spot: float, strike: float, sigma: float, maturity: float,
                       rate: float) -> float:
    """Black-Scholes European call price (no dividends), for put-call parity checks.

    Args:
        spot: underlying price.
        strike: strike price.
        sigma: annualised volatility.
        maturity: time to expiry in years.
        rate: annual continuously-compounded risk-free rate.
    Returns:
        The call price; its intrinsic value if maturity or sigma is non-positive.
    """
    if maturity <= 0 or sigma <= 0:
        return max(spot - strike, 0.0)
    d1 = (np.log(spot / strike) + (rate + 0.5 * sigma ** 2) * maturity) / (sigma * np.sqrt(maturity))
    d2 = d1 - sigma * np.sqrt(maturity)
    return float(spot * norm.cdf(d1) - strike * np.exp(-rate * maturity) * norm.cdf(d2))


def _roll_blocks(dates: pd.DatetimeIndex, tenor_months: int) -> list[pd.DatetimeIndex]:
    """Split a daily index into consecutive blocks of `tenor_months` calendar months.

    Args:
        dates: daily DatetimeIndex.
        tenor_months: option tenor in months (roll at the end of each block).
    Returns:
        List of DatetimeIndex blocks, one per option life.
    """
    periods = dates.to_period("M")
    ordinal = periods.year * 12 + (periods.month - 1)
    block_id = (ordinal - ordinal[0]) // tenor_months
    return [dates[block_id == b] for b in np.unique(block_id)]


def put_overlay_returns(underlying: pd.Series, vix: pd.Series, rate_daily: pd.Series,
                        multiplier: float, hedge_notional: float,
                        tenor_months: int = 1, otm: float = config.PUT_OTM,
                        markup: float = config.PUT_BID_ASK_MARKUP) -> pd.Series:
    """Daily P&L (fraction of portfolio) of a rolled OTM protective-put overlay.

    Each option life buys a `otm`-out-of-the-money put sized to `hedge_notional`,
    priced with sigma = multiplier x VIX/100, marked to market daily and expired at
    intrinsic value. A bid-ask markup is charged on the premium at purchase.

    Args:
        underlying: daily index level used as the option underlying.
        vix: daily VIX level (percent) matching the option tenor.
        rate_daily: daily simple risk-free rate.
        multiplier: VIX-to-sigma scalar m.
        hedge_notional: overlay size as a fraction of portfolio value.
        tenor_months: option tenor in months.
        otm: out-of-the-money fraction (0.05 = 5%).
        markup: bid-ask premium markup.
    Returns:
        Daily overlay P&L as a fraction of portfolio value.
    """
    data = pd.concat([underlying.rename("S"), vix.rename("vix"),
                      rate_daily.rename("rf")], axis=1).dropna()
    annual_rate = np.log1p(data["rf"] * config.TRADING_DAYS_PER_YEAR)
    pnl = pd.Series(0.0, index=data.index)
    for block in _roll_blocks(data.index, tenor_months):
        spot0 = data["S"].iloc[data.index.get_indexer([block[0]])[0]]
        strike = (1.0 - otm) * spot0
        n_days = len(block)
        prices = []
        for i, day in enumerate(block):
            maturity = max((n_days - i) / config.TRADING_DAYS_PER_YEAR, 1e-6)
            sigma = multiplier * data.loc[day, "vix"] / 100.0
            prices.append(black_scholes_put(data.loc[day, "S"], strike, sigma,
                                            maturity, annual_rate.loc[day]))
        prices[-1] = max(strike - data.loc[block[-1], "S"], 0.0)   # expiry intrinsic
        premium = prices[0]
        daily = np.diff(prices, prepend=prices[0]) / spot0 * hedge_notional
        daily[0] = -markup * premium / spot0 * hedge_notional       # bid-ask paid upfront
        pnl.loc[block] = daily
    return pnl.rename(f"put_{tenor_months}m")


def calibrate_put_multiplier(underlying: pd.Series, vix: pd.Series, rate_daily: pd.Series,
                             pput: pd.Series) -> dict[str, float]:
    """Calibrate the VIX multiplier m so index+monthly-put tracks PPUT, then validate.

    The modelled series is the underlying's monthly return plus the monthly put overlay
    (full notional). m is fit by least squares on the first half of overlapping months
    and the tracking error is reported on the held-out second half.

    Args:
        underlying: daily index level (index + put underlying).
        vix: daily 1-month VIX level (percent).
        rate_daily: daily simple risk-free rate.
        pput: daily Cboe PPUT benchmark level.
    Returns:
        Dict with the fitted multiplier and in/out-of-sample monthly tracking errors.
    """
    index_monthly = underlying.resample("ME").last().pct_change().dropna()
    pput_monthly = pput.resample("ME").last().pct_change().dropna()
    common = index_monthly.index.intersection(pput_monthly.index)
    split = common[len(common) // 2]

    def model_monthly(m: float) -> pd.Series:
        overlay = put_overlay_returns(underlying, vix, rate_daily, m, hedge_notional=1.0)
        monthly = overlay.resample("ME").sum()
        return (index_monthly + monthly).reindex(common).dropna()

    def sse(m: float) -> float:
        diff = (model_monthly(m) - pput_monthly.reindex(common)).loc[:split].dropna()
        return float((diff ** 2).sum())

    best = minimize_scalar(sse, bounds=(0.3, 3.0), method="bounded")
    fitted = float(best.x)
    error = (model_monthly(fitted) - pput_monthly.reindex(common)).dropna()
    return {
        "multiplier": fitted,
        "tracking_error_in": float(error.loc[:split].std(ddof=1)),
        "tracking_error_out": float(error.loc[split:].std(ddof=1)),
    }
