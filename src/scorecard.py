"""Net-return scorecard, cost-of-protection metrics and the policy table (section 3.6).

For every dial level x strategy we measure performance net of costs, the carry given up
in calm months, the cost per point of tail-risk removed, and the break-even crash
probability that would justify the protection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config, metrics


def stress_mask(index: pd.DatetimeIndex,
                episodes: list[tuple[str, str, str]] | None = None) -> pd.Series:
    """Boolean mask marking dates that fall inside any stress episode.

    Args:
        index: the daily DatetimeIndex to mark.
        episodes: list of (label, start, end); defaults to config.STRESS_EPISODES.
    Returns:
        Boolean Series, True on stress-episode dates.
    """
    episodes = episodes if episodes is not None else config.STRESS_EPISODES
    mask = pd.Series(False, index=index)
    for _, start, end in episodes:
        mask.loc[start:end] = True
    return mask


def worst_month(returns: pd.Series) -> float:
    """Worst calendar-month compounded return.

    Args:
        returns: daily simple returns.
    Returns:
        The most negative monthly compounded return.
    """
    monthly = returns.resample("ME").apply(lambda window: (1.0 + window).prod() - 1.0)
    return float(monthly.min()) if not monthly.empty else float("nan")


def carry_cost(hedged: pd.Series, unhedged: pd.Series,
               episodes: list[tuple[str, str, str]] | None = None) -> float:
    """Annualised return given up versus unhedged in non-stress (calm) months.

    Args:
        hedged: daily hedged returns.
        unhedged: daily unhedged returns.
        episodes: stress episodes to exclude; defaults to config.STRESS_EPISODES.
    Returns:
        Annualised carry cost (positive means the hedge costs return in calm times).
    """
    calm = ~stress_mask(hedged.index, episodes)
    daily_drag = (unhedged.reindex(hedged.index) - hedged)[calm].mean()
    return float(daily_drag * config.TRADING_DAYS_PER_YEAR)


def average_loss_avoided(hedged: pd.Series, unhedged: pd.Series,
                         episodes: list[tuple[str, str, str]] | None = None) -> float:
    """Average extra return the hedge preserved across stress episodes.

    Args:
        hedged: daily hedged returns.
        unhedged: daily unhedged returns.
        episodes: stress episodes; defaults to config.STRESS_EPISODES.
    Returns:
        Mean over episodes of (hedged minus unhedged) compounded episode return.
    """
    episodes = episodes if episodes is not None else config.STRESS_EPISODES
    avoided = []
    for _, start, end in episodes:
        hedged_ret = metrics.cumulative_return(hedged, start, end)
        unhedged_ret = metrics.cumulative_return(unhedged, start, end)
        if not (np.isnan(hedged_ret) or np.isnan(unhedged_ret)):
            avoided.append(hedged_ret - unhedged_ret)
    return float(np.mean(avoided)) if avoided else float("nan")


def strategy_row(hedged: pd.Series, unhedged: pd.Series, risk_free: pd.Series,
                 episodes: list[tuple[str, str, str]] | None = None) -> dict[str, float]:
    """One scorecard row: performance, costs and break-even for a hedged strategy.

    Args:
        hedged: daily hedged returns.
        unhedged: daily unhedged (dial) returns.
        risk_free: daily simple risk-free rate.
        episodes: stress episodes; defaults to config.STRESS_EPISODES.
    Returns:
        Dict of net performance, ES95 reduction, carry cost, cost per point of ES
        removed, average loss avoided and break-even crash probability.
    """
    ppy = config.TRADING_DAYS_PER_YEAR
    row = {
        "ann_return": metrics.cagr(hedged, ppy),
        "ann_vol": metrics.annualised_volatility(hedged, ppy),
        "sharpe": metrics.sharpe_ratio(hedged, ppy, risk_free),
        "max_drawdown": metrics.max_drawdown(hedged),
        "ES95": metrics.expected_shortfall(hedged, 0.95),
        "worst_month": worst_month(hedged),
    }
    es_reduction_pp = (row["ES95"] - metrics.expected_shortfall(unhedged, 0.95)) * 100.0
    cost = carry_cost(hedged, unhedged, episodes)
    loss_avoided = average_loss_avoided(hedged, unhedged, episodes)
    row["carry_cost"] = cost
    row["ES95_reduction_pp"] = es_reduction_pp
    row["cost_per_pp_ES"] = cost / es_reduction_pp if es_reduction_pp > 1e-9 else float("nan")
    row["avg_loss_avoided"] = loss_avoided
    denominator = loss_avoided + cost
    row["breakeven_crash_prob"] = cost / denominator if denominator > 1e-9 else float("nan")
    return row


def build_scorecard(strategies: dict[tuple, pd.Series], unhedged_by_dial: dict[str, pd.Series],
                    risk_free: pd.Series,
                    episodes: list[tuple[str, str, str]] | None = None) -> pd.DataFrame:
    """Assemble the full (dial x strategy) scorecard.

    Args:
        strategies: map of (dial_label, strategy_name) -> daily hedged returns.
        unhedged_by_dial: map of dial_label -> daily unhedged returns.
        risk_free: daily simple risk-free rate.
        episodes: stress episodes; defaults to config.STRESS_EPISODES.
    Returns:
        DataFrame indexed by (dial, strategy) with the scorecard metrics.
    """
    rows = {key: strategy_row(series, unhedged_by_dial[key[0]], risk_free, episodes)
            for key, series in strategies.items()}
    table = pd.DataFrame(rows).T
    table.index.names = ["dial", "strategy"]
    return table


def policy_table(scorecard: pd.DataFrame) -> pd.DataFrame:
    """Pick, per dial level, the strategy with the most ES removed per unit of cost.

    Args:
        scorecard: a (dial x strategy) scorecard from build_scorecard.
    Returns:
        DataFrame indexed by dial with the chosen strategy and its efficiency, carry
        cost and ES reduction.
    """
    efficiency = scorecard["ES95_reduction_pp"] / scorecard["carry_cost"].abs()
    best_rows = []
    for dial in scorecard.index.get_level_values("dial").unique():
        block = efficiency.loc[dial].replace([np.inf, -np.inf], np.nan).dropna()
        mask = (scorecard.loc[dial, "ES95_reduction_pp"] > 0).reindex(block.index).fillna(False)
        positive = block[mask]
        if positive.empty:
            best_rows.append({"dial": dial, "best_strategy": "do nothing",
                              "es_per_cost": float("nan")})
            continue
        best = positive.idxmax()
        best_rows.append({
            "dial": dial, "best_strategy": best, "es_per_cost": float(positive.max()),
            "carry_cost": float(scorecard.loc[(dial, best), "carry_cost"]),
            "ES95_reduction_pp": float(scorecard.loc[(dial, best), "ES95_reduction_pp"]),
        })
    return pd.DataFrame(best_rows).set_index("dial")
