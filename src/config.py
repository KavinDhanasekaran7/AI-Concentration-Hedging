"""Central configuration: every constant for the project lives here.

No magic numbers belong anywhere else in the codebase. Paths are derived from the
repo root with pathlib so the project is portable across machines.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths (all relative to the repo root, never absolute)
# --------------------------------------------------------------------------- #
REPO_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = REPO_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"            # git-ignored download cache
PROCESSED_DIR: Path = DATA_DIR / "processed"
RESULTS_DIR: Path = REPO_ROOT / "results"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
TABLES_DIR: Path = RESULTS_DIR / "tables"

for _directory in (RAW_DIR, PROCESSED_DIR, FIGURES_DIR, TABLES_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_SEED: int = 42

# --------------------------------------------------------------------------- #
# Sample windows
# --------------------------------------------------------------------------- #
REPLICATION_START: str = "1926-07-01"   # Ken French 49-industry coverage start
STOCK_ANALYSIS_START: str = "2010-01-01"  # EDGAR XBRL coverage is reliable from ~2010
AI_BASKET_START: str = "2015-01-01"       # fixed-basket robustness run

# --------------------------------------------------------------------------- #
# Candidate mega-cap pool (all still listed). The ranked top 10 is drawn from this.
# --------------------------------------------------------------------------- #
CANDIDATE_TICKERS: list[str] = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "AVGO", "TSLA",
    "BRK-B", "LLY", "JPM", "V", "MA", "UNH", "XOM", "CVX", "JNJ", "PG",
    "WMT", "HD", "KO", "PFE", "T", "VZ", "INTC", "CSCO", "ORCL", "IBM",
    "GE", "WFC", "BAC", "MRK",
]

# Fixed AI basket for the hindsight-flagged robustness version.
AI_BASKET_TICKERS: list[str] = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "AVGO"]

# "Rest of market" sleeve and index / hedge instruments.
REST_OF_MARKET_TICKER: str = "RSP"        # equal-weight S&P 500 ETF
INDEX_TICKERS: list[str] = ["SPY", "^SP500TR", "QQQ", "RSP"]
PPUT_TICKER: str = "^PPUT"                # Cboe 5% Put Protection Index (validation)

# Volatility term structure and skew (nearest-tenor fallback handled in loaders).
VIX_TICKERS: list[str] = ["^VIX9D", "^VIX", "^VIX3M", "^VIX6M", "^VIX1Y", "^SKEW"]

# Number of names in the top sleeve each year.
TOP_SLEEVE_SIZE: int = 10

# --------------------------------------------------------------------------- #
# The concentration dial: top-sleeve weight blended with the rest-of-market sleeve
# --------------------------------------------------------------------------- #
DIAL_LEVELS: tuple[float, ...] = (0.20, 0.30, 0.40, 0.50)
DIAL_MARKET_LEVEL: float = 0.40           # labelled "≈ today's US market" in charts

# --------------------------------------------------------------------------- #
# Rebalancing rules
# --------------------------------------------------------------------------- #
SINGLE_NAME_CAP: float = 0.05             # no single stock above 5% of the portfolio
N_CLUSTERS: int = 3                       # clustered inverse-vol cut
CORRELATION_WINDOW_DAYS: int = 252        # 1 trading year

# --------------------------------------------------------------------------- #
# Hedges and costs
# --------------------------------------------------------------------------- #
HEDGE_NOTIONAL: float = 0.30              # common overlay notional (30% of portfolio)
PUT_OTM: float = 0.05                     # 5% out-of-the-money
PUT_BID_ASK_MARKUP: float = 0.05          # 5% premium markup for bid-ask
PUT_LADDER_TENORS_MONTHS: tuple[int, ...] = (1, 3, 6, 12)
TREND_MA_DAYS: int = 200                  # trend overlay moving-average window
CONCENTRATION_HEDGE_FLOOR: float = 0.20   # max(0, w_top - 0.20) sizes the conc. hedge
BETA_WINDOW_DAYS: int = 252               # trailing beta of top sleeve on QQQ
OVERLAY_TURNOVER_COST: float = 0.0002     # 2 bps per unit of overlay turnover
SHORT_BORROW_SPREAD: float = 0.0025       # shorts earn T-bill minus 25 bps

# CVaR optimisation (Rockafellar-Uryasev LP).
CVAR_ALPHA: float = 0.95                  # CVaR95 on weekly returns
CVAR_ESTIMATION_YEARS: int = 5            # trailing window for walk-forward
CVAR_MIN_RETURN_GIVEUP: float = 0.01      # mean >= unhedged mean - 1%/yr

# Risk-measure confidence levels reported in the scorecard.
VAR_LEVELS: tuple[float, ...] = (0.95, 0.975)

# --------------------------------------------------------------------------- #
# Stress episodes (defined once here; (label, start, end) inclusive)
# --------------------------------------------------------------------------- #
STRESS_EPISODES: list[tuple[str, str, str]] = [
    ("2011 Aug-Oct selloff", "2011-08-01", "2011-10-31"),
    ("2015 Aug selloff", "2015-08-01", "2015-08-31"),
    ("2018 Q4 selloff", "2018-10-01", "2018-12-31"),
    ("2020 COVID crash", "2020-02-19", "2020-03-23"),
    ("2022 bear market", "2022-01-01", "2022-12-31"),
    ("2025 Feb-Apr selloff", "2025-02-19", "2025-04-08"),
    ("2025-26 software selloff", "2025-10-01", "2026-02-28"),
]

TRADING_DAYS_PER_YEAR: int = 252
WEEKS_PER_YEAR: int = 52
MONTHS_PER_YEAR: int = 12

# --------------------------------------------------------------------------- #
# Data-source URLs
# --------------------------------------------------------------------------- #
KEN_FRENCH_INDUSTRY_URL: str = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "49_Industry_Portfolios_CSV.zip"
)
KEN_FRENCH_FACTORS_URL: str = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_CSV.zip"
)
SEC_TICKER_MAP_URL: str = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANY_FACTS_URL: str = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
CBOE_HISTORY_URL: str = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{name}_History.csv"
FRED_DTB3_URL: str = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3"

# Missing-value sentinels in the Ken French files.
KEN_FRENCH_MISSING: tuple[float, ...] = (-99.99, -999.0)

# --------------------------------------------------------------------------- #
# SEC EDGAR access
# --------------------------------------------------------------------------- #
# EDGAR requires a descriptive User-Agent. Kavin fills in the real name/email.
SEC_USER_AGENT: str = "Your Name your.email@example.com"
SEC_REQUEST_SLEEP_SECONDS: float = 0.15   # be polite: >= 0.15 s between requests
