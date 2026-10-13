"""Data loaders with on-disk caching for every source used in the project.

Raw downloads are cached under data/raw/ (git-ignored); nothing re-downloads unless
force=True. Each public loader returns a tidy pandas object with a DatetimeIndex.
If a download fails the loader raises with a clear message rather than inventing data.
"""
from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

from src import config


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #
def _download_bytes(url: str, headers: dict | None = None, timeout: int = 60) -> bytes:
    """Download a URL and return its raw bytes, raising on any HTTP error.

    Args:
        url: the resource to fetch.
        headers: optional request headers (e.g. a User-Agent for SEC/French).
        timeout: seconds before giving up.
    Returns:
        The response body as bytes.
    """
    response = requests.get(url, headers=headers or {}, timeout=timeout)
    response.raise_for_status()
    return response.content


def _cache_csv(path: Path, build, force: bool) -> pd.DataFrame:
    """Return a cached CSV if present, else build it, save it and return it.

    Args:
        path: processed-CSV cache location.
        build: zero-arg callable returning a DataFrame with a DatetimeIndex.
        force: re-build and overwrite the cache when True.
    Returns:
        The cached or freshly built DataFrame.
    """
    if path.exists() and not force:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    frame = build()
    frame.to_csv(path)
    return frame


# --------------------------------------------------------------------------- #
# Ken French Data Library (49 industries + factors)
# --------------------------------------------------------------------------- #
def _french_zip_text(url: str, raw_name: str, force: bool) -> str:
    """Download (and cache) a Ken French CSV zip and return the inner CSV text.

    Args:
        url: the French zip URL.
        raw_name: filename to cache the raw CSV under in data/raw/.
        force: re-download when True.
    Returns:
        The decoded text of the single CSV inside the zip.
    """
    raw_path = config.RAW_DIR / raw_name
    if raw_path.exists() and not force:
        return raw_path.read_text()
    content = _download_bytes(url, headers={"User-Agent": config.SEC_USER_AGENT})
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        inner_name = archive.namelist()[0]
        text = archive.read(inner_name).decode("latin-1")
    raw_path.write_text(text, encoding="utf-8", newline="")  # no newline translation
    return text


def _parse_french_section(text: str, marker: str, index_len: int,
                          header_offset: int = 1) -> pd.DataFrame:
    """Parse one stacked table from a Ken French CSV by a marker line.

    Args:
        text: full CSV text.
        marker: substring identifying the section. With header_offset=1 it names
            a title line sitting one row above the column header (the industry
            tables); with header_offset=0 it names the column header itself (the
            factors file, whose header row contains 'Mkt-RF').
        index_len: 6 for monthly (YYYYMM) rows, 4 for annual (YYYY) rows.
        header_offset: rows from the marked line down to the column-header line.
    Returns:
        DataFrame indexed by period-end timestamp, columns = series codes,
        French missing sentinels replaced by NaN.
    """
    lines = text.splitlines()
    marker_idx = next(i for i, line in enumerate(lines) if marker in line)
    if header_offset == 0:
        start = marker_idx                       # the marked line is the column header
    else:
        start = next(i for i in range(marker_idx + 1, len(lines)) if lines[i].strip())
    header = lines[start].split(",")
    columns = [name.strip() for name in header[1:] if name.strip()]
    records: dict = {}
    for line in lines[start + 1:]:
        token = line.split(",")[0].strip()
        if token.isdigit() and len(token) == index_len:
            values = [float(v) for v in line.split(",")[1: len(columns) + 1]]
            records[_french_stamp(token)] = values
        elif token == "":
            continue          # blank line: French interleaves these between rows
        elif records:
            break             # a new section title after data ends this section
    frame = pd.DataFrame.from_dict(records, orient="index", columns=columns)
    frame.index = pd.DatetimeIndex(frame.index, name="date")
    return frame.replace(list(config.KEN_FRENCH_MISSING), float("nan"))


def _french_stamp(token: str) -> pd.Timestamp:
    """Convert a French YYYYMM or YYYY token to a period-end Timestamp."""
    if len(token) == 6:
        return pd.Period(f"{token[:4]}-{token[4:]}", freq="M").end_time.normalize()
    return pd.Period(token, freq="Y").end_time.normalize()


def load_industry_portfolios(force: bool = False) -> dict[str, pd.DataFrame]:
    """Load the 49-industry monthly returns, firm counts and average firm size.

    Args:
        force: re-download and re-parse when True.
    Returns:
        Dict with keys 'returns' (value-weighted %, monthly), 'n_firms' and
        'avg_size', each a DataFrame indexed by month-end, columns = industries.
    """
    text = _french_zip_text(config.KEN_FRENCH_INDUSTRY_URL, "french_49_industry.csv", force)
    return {
        "returns": _parse_french_section(text, "Average Value Weighted Returns -- Monthly", 6),
        "n_firms": _parse_french_section(text, "Number of Firms in Portfolios", 6),
        "avg_size": _parse_french_section(text, "Average Firm Size", 6),
    }


def load_ff_factors(force: bool = False) -> pd.DataFrame:
    """Load the Fama-French monthly market, risk-free and excess-return factors.

    Args:
        force: re-download and re-parse when True.
    Returns:
        DataFrame indexed by month-end with columns Mkt-RF, SMB, HML, RF and
        Mkt (= Mkt-RF + RF), all in percent per month.
    """
    def build() -> pd.DataFrame:
        text = _french_zip_text(config.KEN_FRENCH_FACTORS_URL, "french_ff_factors.csv", force)
        frame = _parse_french_section(text, "Mkt-RF", 6, header_offset=0)
        frame.columns = [c.strip() for c in frame.columns]
        frame["Mkt"] = frame["Mkt-RF"] + frame["RF"]
        return frame

    return _cache_csv(config.PROCESSED_DIR / "ff_factors.csv", build, force)


# --------------------------------------------------------------------------- #
# yfinance daily prices
# --------------------------------------------------------------------------- #
def load_prices(tickers: list[str], name: str, start: str | None = None,
                force: bool = False) -> pd.DataFrame:
    """Download (and cache) daily auto-adjusted close prices for a ticker list.

    Args:
        tickers: Yahoo tickers to download.
        name: cache filename stem (one CSV per logical group).
        start: optional ISO start date; None uses each ticker's full history.
        force: re-download when True.
    Returns:
        DataFrame of adjusted closes indexed by date, one column per ticker,
        columns ordered as requested (tickers with no data are dropped).
    """
    def build() -> pd.DataFrame:
        # period='max' only when no start given: yfinance otherwise defaults to 1 month.
        kwargs = {"start": start} if start else {"period": "max"}
        raw = yf.download(tickers, auto_adjust=True, progress=False, **kwargs)
        closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        if not isinstance(raw.columns, pd.MultiIndex):
            closes.columns = [tickers[0]]
        closes = closes.reindex(columns=[t for t in tickers if t in closes.columns])
        closes.index.name = "date"
        if closes.dropna(how="all").empty:
            raise RuntimeError(f"No price data returned for {name}: {tickers}")
        return closes.dropna(how="all")

    return _cache_csv(config.PROCESSED_DIR / f"prices_{name}.csv", build, force)


def load_pput(force: bool = False) -> pd.Series:
    """Load the Cboe 5% Put Protection Index (^PPUT), Yahoo first then Cboe CSV.

    Args:
        force: re-download when True.
    Returns:
        Series of PPUT closing levels indexed by date. Used only to calibrate and
        validate modelled put costs.
    """
    frame = load_prices([config.PPUT_TICKER], "pput", start="1990-01-01", force=force)
    series = frame.iloc[:, 0] if not frame.empty else pd.Series(dtype=float)
    if len(series) < config.TRADING_DAYS_PER_YEAR:
        print("WARNING: ^PPUT unavailable/short on Yahoo; falling back to Cboe CSV")
        series = load_cboe_index("PPUT", force=force)
    return series.rename("PPUT")


def load_vix_term_structure(force: bool = False) -> pd.DataFrame:
    """Load the VIX term structure and SKEW, with nearest-tenor fallback warnings.

    Each tenor is downloaded separately and outer-joined, because a group download
    truncates every series to the shortest (some Yahoo VIX tenors are very short).

    Args:
        force: re-download when True.
    Returns:
        DataFrame indexed by date with one column per available VIX/SKEW ticker.
    """
    def build() -> pd.DataFrame:
        columns: dict[str, pd.Series] = {}
        for ticker in config.VIX_TICKERS:
            raw = yf.download(ticker, start="1990-01-01", auto_adjust=True, progress=False)
            close = raw["Close"] if "Close" in raw.columns else raw
            series = close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close
            if len(series) < config.TRADING_DAYS_PER_YEAR:
                print(f"WARNING: VIX tenor {ticker} unavailable/short on Yahoo "
                      f"({len(series)} rows); nearest tenor will be used downstream")
                continue
            columns[ticker] = series
        frame = pd.concat(columns, axis=1).sort_index()
        frame.index.name = "date"
        return frame

    return _cache_csv(config.PROCESSED_DIR / "prices_vix.csv", build, force)


# --------------------------------------------------------------------------- #
# Cboe CSV fallback (used when a Yahoo index is unavailable)
# --------------------------------------------------------------------------- #
def load_cboe_index(name: str, force: bool = False) -> pd.Series:
    """Download a Cboe daily-price history CSV and return its close series.

    Args:
        name: Cboe index name used in the CSV URL (e.g. 'VIX', 'PPUT', 'SKEW').
        force: re-download when True.
    Returns:
        Series of closing values indexed by date.
    """
    def build() -> pd.DataFrame:
        url = config.CBOE_HISTORY_URL.format(name=name)
        text = _download_bytes(url, headers={"User-Agent": config.SEC_USER_AGENT}).decode("utf-8")
        frame = pd.read_csv(io.StringIO(text))
        frame.columns = [c.strip().lower() for c in frame.columns]
        date_col, value_col = frame.columns[0], frame.columns[-1]
        frame = frame[[date_col, value_col]].dropna()
        frame[date_col] = pd.to_datetime(frame[date_col])
        return frame.set_index(date_col).rename(columns={value_col: name})

    cached = _cache_csv(config.PROCESSED_DIR / f"cboe_{name}.csv", build, force)
    return cached.iloc[:, 0]


# --------------------------------------------------------------------------- #
# FRED risk-free rate
# --------------------------------------------------------------------------- #
def load_risk_free_daily(force: bool = False) -> pd.Series:
    """Load the 3-month T-bill rate (FRED DTB3) as a daily simple rate.

    Args:
        force: re-download when True.
    Returns:
        Series of daily simple risk-free rates indexed by date (annual %/100/252).
    """
    def build() -> pd.DataFrame:
        text = _download_bytes(config.FRED_DTB3_URL).decode("utf-8")
        frame = pd.read_csv(io.StringIO(text))
        frame.columns = ["date", "DTB3"]
        frame["date"] = pd.to_datetime(frame["date"])
        frame["DTB3"] = pd.to_numeric(frame["DTB3"], errors="coerce")
        frame = frame.dropna().set_index("date")
        frame["rf_daily"] = frame["DTB3"] / 100.0 / config.TRADING_DAYS_PER_YEAR
        return frame[["rf_daily"]]

    return _cache_csv(config.PROCESSED_DIR / "rf_daily.csv", build, force).iloc[:, 0]


# --------------------------------------------------------------------------- #
# SEC EDGAR shares outstanding
# --------------------------------------------------------------------------- #
def _load_cik_map(force: bool = False) -> dict[str, int]:
    """Download (and cache) the SEC ticker->CIK map.

    Args:
        force: re-download when True.
    Returns:
        Dict mapping upper-case ticker to integer CIK.
    """
    raw_path = config.RAW_DIR / "sec_company_tickers.json"
    if not raw_path.exists() or force:
        content = _download_bytes(config.SEC_TICKER_MAP_URL,
                                  headers={"User-Agent": config.SEC_USER_AGENT})
        raw_path.write_bytes(content)
    table = pd.read_json(raw_path, orient="index")
    return {row["ticker"].upper(): int(row["cik_str"]) for _, row in table.iterrows()}


def _shares_from_facts(facts: dict) -> pd.Series:
    """Extract a dated shares-outstanding series from one EDGAR companyfacts blob.

    Tags are tried in priority order. The weighted-average basic-shares fallback
    rescues multi-class firms (e.g. Meta) whose cover-page count is filed per class
    and so absent from the single EntityCommonStockSharesOutstanding tag.

    Args:
        facts: parsed companyfacts JSON for one company.
    Returns:
        Series of shares outstanding indexed by report 'end' date (may be empty).
    """
    candidate_tags = (
        ("dei", "EntityCommonStockSharesOutstanding"),
        ("us-gaap", "CommonStockSharesOutstanding"),
        ("us-gaap", "WeightedAverageNumberOfSharesOutstandingBasic"),
    )
    for taxonomy, tag in candidate_tags:
        node = facts.get("facts", {}).get(taxonomy, {}).get(tag)
        if not node:
            continue
        points = next(iter(node["units"].values()))
        series = pd.Series({pd.Timestamp(p["end"]): float(p["val"]) for p in points})
        return series.sort_index()
    return pd.Series(dtype=float)


def load_shares_outstanding(tickers: list[str], force: bool = False) -> pd.DataFrame:
    """Load year-end shares outstanding for each ticker from SEC EDGAR XBRL.

    Args:
        tickers: tickers to fetch (Yahoo style; '-' is mapped to SEC '.').
        force: re-download when True.
    Returns:
        DataFrame indexed by date with one column per ticker of reported shares.
    """
    def build() -> pd.DataFrame:
        cik_map = _load_cik_map(force)
        columns: dict[str, pd.Series] = {}
        for ticker in tickers:
            sec_ticker = ticker.replace("-", ".").upper()
            cik = cik_map.get(sec_ticker) or cik_map.get(ticker.upper())
            if cik is None:
                print(f"WARNING: no CIK for {ticker}; skipping shares history")
                continue
            url = config.SEC_COMPANY_FACTS_URL.format(cik=cik)
            facts = requests.get(url, headers={"User-Agent": config.SEC_USER_AGENT},
                                 timeout=60).json()
            columns[ticker] = _shares_from_facts(facts)
            time.sleep(config.SEC_REQUEST_SLEEP_SECONDS)
        frame = pd.DataFrame(columns).sort_index()
        frame.index.name = "date"
        if frame.empty:
            raise RuntimeError("EDGAR returned no shares data for any ticker")
        return frame

    return _cache_csv(config.PROCESSED_DIR / "shares_outstanding.csv", build, force)
