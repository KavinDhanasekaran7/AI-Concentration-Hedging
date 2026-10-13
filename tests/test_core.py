"""Core unit tests. Small, fast and deterministic.

Fuller tests (metrics, ES, Black-Scholes, CVaR LP, etc.) are added as each module
lands; for now this verifies the config wiring so `pytest` runs from Stage 0.
"""
from __future__ import annotations

from src import config


def test_dial_levels_sorted_and_in_unit_interval() -> None:
    """Dial levels are ascending fractions strictly inside (0, 1)."""
    levels = config.DIAL_LEVELS
    assert list(levels) == sorted(levels)
    assert all(0.0 < level < 1.0 for level in levels)


def test_market_dial_level_is_a_dial_level() -> None:
    """The level labelled as today's market must be one of the dial levels."""
    assert config.DIAL_MARKET_LEVEL in config.DIAL_LEVELS


def test_candidate_pool_has_no_duplicates() -> None:
    """The candidate mega-cap pool contains no repeated tickers."""
    assert len(config.CANDIDATE_TICKERS) == len(set(config.CANDIDATE_TICKERS))


def test_required_directories_exist() -> None:
    """Importing config creates the data and results directory tree."""
    for directory in (config.RAW_DIR, config.PROCESSED_DIR, config.FIGURES_DIR, config.TABLES_DIR):
        assert directory.is_dir()
