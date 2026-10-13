# CLAUDE.md

Guidance for Claude Code when working in this project. The full build spec lives in
`../details.txt` (the parent `Own project/` folder — kept out of the public repo) —
read it before any non-trivial step; this file is the distilled, operational version of it.

## Project

**Title:** When AI Stocks Are the Market: Managing AI-Driven Concentration Risk and Hedging Its Tail

**Author:** Kavin Dhanasekaran — MS Financial Engineering, NYU Tandon (linkedin.com/in/kavnd)

A **public GitHub portfolio project**. Build it end to end: data download → analysis →
figures → README. Audience is recruiters and professors: every function must be
understandable in under a minute.

**Central question:** Does market concentration change a portfolio's *tail risk* even when
it does not predict returns — and at what concentration level should a manager do nothing,
rebalance, or hedge, for the best net return per unit of protection bought?

**Hypotheses (confirm or reject honestly — never tune a method after seeing results):**
- **H1** Concentration doesn't predict returns but worsens the tail (more ES from top names, falling effective bets in stress).
- **H2** The put+trend blend protects a concentrated portfolio less per dollar than a diversified one; a concentration hedge closes the gap.
- **H3** The walk-forward CVaR-optimal mix leans more on the concentration hedge as concentration rises.
- **H4** At moderate concentration, rebalancing wins on net return per unit of tail risk removed; at high concentration, overlay hedges do.

## Repository layout (target)

Build inside a repo root named `ai-concentration-tail-hedging/`:

```
README.md  CLAUDE.md  requirements.txt  .gitignore  LICENSE (MIT)
src/   config.py data_loaders.py replication.py portfolio.py concentration.py
       rebalancing.py options.py hedges.py cvar.py metrics.py scorecard.py plots.py __init__.py
notebooks/ai_concentration_tail_hedging.ipynb
tests/test_core.py
data/raw (git-ignored)   data/processed
results/figures (PNG 150dpi)   results/tables (CSV)
```

## Code style (non-negotiable)

- Plain functions, no classes unless clearly simpler. Aim ≤ 40 lines/function; split anything > 150 lines.
- Every function: one-line docstring + Args/Returns + type hints on the signature.
- Descriptive names (`expected_shortfall`, not `es_fn2`). No clever one-liners.
- **All constants** (dates, tickers, dial levels `{0.20,0.30,0.40,0.50}`, costs, stress episodes, seeds, `SEC_USER_AGENT`) live in `config.py`. No magic numbers elsewhere.
- Paths via `pathlib`, relative to repo root. No absolute paths.
- `pandas` `DatetimeIndex` everywhere; daily returns are simple returns unless stated. Vectorise with pandas/numpy.
- Set numpy seeds in `config.py` wherever randomness exists.
- **Dependencies only:** pandas, numpy, scipy, statsmodels, matplotlib, yfinance, requests, jupyter, nbconvert, pytest. **No cvxpy, no seaborn**, no plotting beyond matplotlib.
- Figures: one clear message each, axes labelled with units, titles that state the finding, colours defined once in `plots.py`. Save every figure and table to `results/`.

## Data (all free — no paid data, no index-constituent history)

Cache raw downloads in `data/raw/` (git-ignored), processed series in `data/processed/` as CSV.
Re-running must **not** re-download unless `force=True`. See `details.txt` §2 for every source
(Ken French 49 industries + factors, yfinance mega-caps/ETFs/VIX term structure, SEC EDGAR XBRL
shares, FRED DTB3, Cboe fallbacks). **If any download fails, stop and report which and why —
never invent, interpolate across long gaps, or hard-code data.** EDGAR needs a `User-Agent`
from `config.SEC_USER_AGENT` (leave placeholder for Kavin to fill), sleep ≥ 0.15s between requests.

## Notebook

Single notebook `notebooks/ai_concentration_tail_hedging.ipynb`: minimal markdown (each section =
one 1–3 sentence markdown cell + a few code cells calling `src/` functions). No long logic in the
notebook. Must run top-to-bottom in < 10 min after data is cached. Execute with:

```
jupyter nbconvert --to notebook --execute --inplace notebooks/ai_concentration_tail_hedging.ipynb
```

Commit it with outputs.

## Build in stages — commit after each, report a 3–5 line summary

| Stage | Build | Done when |
|---|---|---|
| 0 | Repo skeleton, config.py, requirements.txt, .gitignore, git init | pytest runs (even with no tests) |
| 1 | data_loaders.py + caching; notebook §1 | all series load; print start/end; report any failure |
| 2 | Replication §3.1; notebook §2 | regression + timing-rule tables saved |
| 3 | Dial portfolios + tail structure §3.2; notebook §3 | H1 figures + tables saved |
| 4 | Rebalancing rules §3.3; notebook §4 | table saved |
| 5 | Monthly puts + PPUT calibration, trend, concentration hedge, scorecard; notebook §5–6, first §8 | minimum viable result |
| 6 | Put ladder, naive hedge, CVaR walk-forward; notebook §7, full §8 | full scorecard + policy table |
| 7 | Robustness §9, tests, README, final notebook run | all tests pass; notebook runs clean top to bottom |

Clear commit messages (e.g. "Stage 3: concentration dial and tail-structure metrics").
**Do not push** — at the end, print the exact `git`/`gh` commands for Kavin to create the repo and push.

## Tests (`tests/test_core.py`)

Small, fast, deterministic: max drawdown on a known series; VaR/ES on a known distribution;
effective N for equal weights = N; ENB for identical eigenvalues = #assets; Black–Scholes put vs
a textbook value; put–call parity; CVaR LP on a 3-asset toy returning weights summing ≤ 1 with
lower CVaR than any single asset.

## README

Portfolio-ready, **generated from computed results — never hard-code numbers not computed.**
Must include: pitch, 4 key-finding bullets (one per hypothesis with real numbers), a headline
tail-risk-vs-concentration figure + a net-return chart, motivation, papers-extended table, data
sources table + survivorship-bias note, methods summary, a **Methodology notes** section listing
every simplification (VIX-based option pricing, RSP contamination, multi-class share counts,
SPY/QQQ as futures proxies, EDGAR coverage from ~2010), how-to-run, limitations, references,
author line, "research only, not investment advice" disclaimer, MIT licence.

## Rules

1. If anything in the spec is ambiguous or a data source is unavailable, **ask Kavin — don't guess.**
2. Keep code simple over clever.
3. Report honest results, including rejected hypotheses.
4. Before finishing, re-read the README and confirm every number in it matches `results/tables/`.
