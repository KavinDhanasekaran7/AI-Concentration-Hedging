"""Core unit tests. Small, fast and deterministic.

Each test checks a numerical primitive against a value known in closed form or by
construction, so a regression in the maths fails loudly.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import hadamard

from src import concentration, config, cvar, metrics, options, replication


# --------------------------------------------------------------------------- #
# Configuration wiring
# --------------------------------------------------------------------------- #
def test_dial_levels_sorted_and_in_unit_interval() -> None:
    """Dial levels are ascending fractions strictly inside (0, 1)."""
    assert list(config.DIAL_LEVELS) == sorted(config.DIAL_LEVELS)
    assert all(0.0 < level < 1.0 for level in config.DIAL_LEVELS)


def test_market_dial_level_is_a_dial_level() -> None:
    """The level labelled as today's market must be one of the dial levels."""
    assert config.DIAL_MARKET_LEVEL in config.DIAL_LEVELS


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_max_drawdown_known_series() -> None:
    """A rise to +25% then fall to -20% gives a known 36% drawdown."""
    returns = pd.Series([0.25, -0.20, -0.20])  # 1.25 -> 1.0 -> 0.80, peak 1.25
    assert metrics.max_drawdown(returns) == pytest.approx(0.80 / 1.25 - 1.0)


def test_var_and_es_on_uniform_tail() -> None:
    """On 0..99, VaR95 is the 5th percentile and ES95 is the mean of the worst 5%."""
    returns = pd.Series(np.arange(100, dtype=float))
    assert metrics.value_at_risk(returns, 0.95) == pytest.approx(returns.quantile(0.05))
    assert metrics.expected_shortfall(returns, 0.95) == pytest.approx(returns[returns <= returns.quantile(0.05)].mean())


def test_expected_shortfall_not_above_var() -> None:
    """Expected shortfall is never less extreme than VaR."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    returns = pd.Series(rng.standard_normal(1000))
    assert metrics.expected_shortfall(returns, 0.95) <= metrics.value_at_risk(returns, 0.95)


# --------------------------------------------------------------------------- #
# Concentration measures
# --------------------------------------------------------------------------- #
def test_effective_number_equal_weights_equals_n() -> None:
    """Effective number of industries for equal weights equals the asset count."""
    weights = pd.DataFrame([[0.25, 0.25, 0.25, 0.25]], columns=list("ABCD"))
    assert replication.effective_number(weights).iloc[0] == pytest.approx(4.0)


def test_enb_equal_eigenvalues_equals_asset_count() -> None:
    """With orthogonal equal-variance contributions, ENB equals the asset count."""
    basis = hadamard(4)[:, 1:]  # three orthogonal, zero-mean, equal-variance columns
    contributions = pd.DataFrame(basis.astype(float), columns=list("XYZ"))
    assert concentration.effective_number_of_bets(contributions) == pytest.approx(3.0)


# --------------------------------------------------------------------------- #
# Options
# --------------------------------------------------------------------------- #
def test_black_scholes_put_textbook_value() -> None:
    """BS put for S=K=100, sigma=0.2, T=1, r=0.05 matches the textbook 5.5735."""
    assert options.black_scholes_put(100, 100, 0.2, 1.0, 0.05) == pytest.approx(5.5735, abs=1e-3)


def test_put_call_parity() -> None:
    """Call minus put equals spot minus discounted strike."""
    call = options.black_scholes_call(100, 95, 0.25, 0.5, 0.03)
    put = options.black_scholes_put(100, 95, 0.25, 0.5, 0.03)
    assert call - put == pytest.approx(100 - 95 * math.exp(-0.03 * 0.5), abs=1e-6)


# --------------------------------------------------------------------------- #
# CVaR linear programme
# --------------------------------------------------------------------------- #
def _cvar95(returns: pd.Series) -> float:
    """Mean of the worst 5% of returns (a negative number)."""
    return returns[returns <= returns.quantile(0.05)].mean()


def test_cvar_lp_diversifies_three_asset_toy() -> None:
    """The CVaR mix sums to <=1 and has CVaR no worse than any single asset."""
    periods = 120
    assets = pd.DataFrame(0.01, index=range(periods), columns=list("ABC"))
    assets.loc[0::3, "A"] = -0.10          # each asset crashes on disjoint weeks
    assets.loc[1::3, "B"] = -0.10
    assets.loc[2::3, "C"] = -0.10
    base = pd.Series(0.0, index=assets.index)

    weights = cvar.solve_cvar_weights(base, assets, giveup_weekly=1.0)
    assert weights.sum() <= 1.0 + 1e-6
    assert (weights >= -1e-9).all()

    mix_cvar = _cvar95((assets * weights).sum(axis=1))
    best_single_cvar = max(_cvar95(assets[name]) for name in assets.columns)
    # CVaR here is the (negative) mean tail return; less negative is less risky. A
    # single-asset allocation is feasible, so the optimum is no worse than the best one.
    assert mix_cvar >= best_single_cvar - 1e-9


def test_required_directories_exist() -> None:
    """Importing config creates the data and results directory tree."""
    for directory in (config.RAW_DIR, config.PROCESSED_DIR, config.FIGURES_DIR, config.TABLES_DIR):
        assert directory.is_dir()
