"""CVaR-optimal hedge mix via the Rockafellar-Uryasev linear programme (section 3.5).

Choose non-negative weights on the hedge overlays to minimise CVaR95 of weekly returns,
subject to the weights summing to at most one and giving up at most 1%/yr of mean return.
Weights are estimated walk-forward on trailing weekly data and applied out of sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from src import config


def weekly_sum(daily: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Aggregate daily P&L to weekly by summing (overlay P&L is additive).

    Args:
        daily: daily returns or overlay P&L.
    Returns:
        Weekly-summed series or frame.
    """
    return daily.resample("W").sum()


def solve_cvar_weights(unhedged: pd.Series, overlays: pd.DataFrame,
                       confidence: float = config.CVAR_ALPHA,
                       giveup_weekly: float = config.CVAR_MIN_RETURN_GIVEUP / config.WEEKS_PER_YEAR
                       ) -> pd.Series:
    """Solve the Rockafellar-Uryasev LP for the CVaR-minimising overlay weights.

    Args:
        unhedged: weekly unhedged portfolio returns (scenarios).
        overlays: weekly overlay returns, one column per hedge at unit notional.
        confidence: CVaR confidence level (0.95).
        giveup_weekly: maximum allowed weekly mean-return give-up versus unhedged.
    Returns:
        Non-negative overlay weights (summing to at most 1), indexed by overlay name.
    """
    scenarios = pd.concat([unhedged.rename("base"), overlays], axis=1).dropna()
    base = scenarios["base"].to_numpy()
    overlay_matrix = scenarios[overlays.columns].to_numpy()
    n_obs, n_hedges = overlay_matrix.shape
    tail_scale = 1.0 / ((1.0 - confidence) * n_obs)

    cost = np.concatenate([np.zeros(n_hedges), [1.0], np.full(n_obs, tail_scale)])
    # z_t + alpha + base_t + overlays_t . x >= 0  ->  -overlays x - alpha - z <= base
    a_tail = np.hstack([-overlay_matrix, -np.ones((n_obs, 1)), -np.eye(n_obs)])
    a_budget = np.concatenate([np.ones(n_hedges), np.zeros(1 + n_obs)])
    a_return = np.concatenate([-overlay_matrix.mean(axis=0), np.zeros(1 + n_obs)])
    a_ub = np.vstack([a_tail, a_budget, a_return])
    b_ub = np.concatenate([base, [1.0], [giveup_weekly]])

    bounds = [(0.0, 1.0)] * n_hedges + [(None, None)] + [(0.0, None)] * n_obs
    result = linprog(cost, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
    weights = result.x[:n_hedges] if result.success else np.zeros(n_hedges)
    return pd.Series(weights, index=overlays.columns)


def walk_forward_cvar(unhedged_daily: pd.Series, overlays_daily: pd.DataFrame,
                      estimation_years: int = config.CVAR_ESTIMATION_YEARS
                      ) -> tuple[pd.DataFrame, pd.Series]:
    """Estimate CVaR weights on trailing weekly data, apply to the next year, roll.

    Args:
        unhedged_daily: daily unhedged portfolio returns.
        overlays_daily: daily overlay returns at unit notional (one column per hedge).
        estimation_years: trailing window length in years.
    Returns:
        Tuple of (yearly weights DataFrame indexed by year, out-of-sample hedged daily
        return series built by applying each year's weights to that year's returns).
    """
    unhedged_weekly = weekly_sum(unhedged_daily)
    overlays_weekly = weekly_sum(overlays_daily)
    first_year = unhedged_daily.index.min().year + estimation_years
    last_year = unhedged_daily.index.max().year

    weights_by_year, hedged_segments = {}, []
    for year in range(first_year, last_year + 1):
        train_start = f"{year - estimation_years}-01-01"
        train = unhedged_weekly.loc[train_start:f"{year - 1}-12-31"]
        overlay_train = overlays_weekly.loc[train_start:f"{year - 1}-12-31"]
        if len(train) < config.WEEKS_PER_YEAR:
            continue
        weights = solve_cvar_weights(train, overlay_train)
        weights_by_year[year] = weights
        applied = overlays_daily.loc[f"{year}-01-01":f"{year}-12-31"].mul(weights, axis=1).sum(axis=1)
        hedged_segments.append(unhedged_daily.loc[f"{year}-01-01":f"{year}-12-31"] + applied)

    weights_frame = pd.DataFrame(weights_by_year).T
    weights_frame.index.name = "year"
    hedged = pd.concat(hedged_segments) if hedged_segments else pd.Series(dtype=float)
    return weights_frame, hedged.rename("cvar_mix")
