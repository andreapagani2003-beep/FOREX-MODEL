# HANDOFF: USD/JPY vs US–Japan Yield Spread, Mean-Reversion Signal Engine

You are Claude Code working in a fresh GitHub repository. This document is the full brief. Read it end to end before writing any code, then copy the "Working agreement" section into `CLAUDE.md` at the repo root.

## 1. Context

Owner: Andrea. Quant background (Applied Economics, Fintech/AI; thesis on Markov-switching regime models for S&P 500 option strategies). Comfortable with statistics, Python, and IBKR. Treat him as a peer, not a beginner.

Strategy under development, as established in the prior session:

- The relationship being exploited is the cointegration between USD/JPY and the US–Japan government bond yield spread. It is NOT "long the currency whose yield rises faster" (that is a carry/momentum idea). It is: FX and the spread share a long-run equilibrium; when FX deviates from the level the spread implies, bet on convergence.
- Model: `USDJPY_t = alpha + beta * spread_t + eps_t`, where `spread_t = US_yield_t - JGB_yield_t`. Trade the residual `eps_t` via a rolling z-score.
- Legs: FX (spot, 6J futures, or CFD) plus, optionally, the spread itself (UST futures vs JGB futures) as the hedge. One-leg (FX only) is directional; two-leg is market-neutral. Build both, compare.
- Timeframe conclusion: daily bars, holding periods of days to a few weeks. Intraday is noise; monthly is macro, not stat-arb. Bar size must match the OU half-life (roughly 5 to 50 bars is tradeable).
- Known regime breaks to test explicitly: BoJ YCC tweak Dec 2022, YCC changes Jul and Oct 2023, NIRP exit Mar 2024, BoJ hike + carry unwind Jul/Aug 2024, MoF FX interventions (Sep/Oct 2022, Apr/May 2024), and any later BoJ/Fed policy events you find in the data range.
- Historical anchors used as sanity checks (approximate): Jan 2023, USD/JPY ~128 with 10y spread ~3.1% (residual strongly negative, converged by May 2023); Jul 2024, USD/JPY ~161 with spread ~3.4% (residual strongly positive, reverted in Aug 2024). Your fitted model should flag both.

## 2. Goal

Build, rigorously and in order:

1. A reproducible data pipeline for US yields, JGB yields, and USD/JPY.
2. Statistical confirmation that the relationship exists, is stable enough to trade, and on which specification (tenor, nominal vs real, static vs rolling beta).
3. A backtest with realistic costs and walk-forward validation.
4. A daily signal engine that outputs a versioned, machine-readable signal.
5. Export of that signal to IBKR TWS and MetaTrader 5.

Do not skip ahead. Each phase has acceptance criteria and ends with a commit, a short entry in `RESEARCH_LOG.md`, and a stop for review. If a phase fails its acceptance criteria, report that and stop; do not "fix" it by loosening the criteria.

## 3. Repository layout

```
usdjpy-spread-mr/
  CLAUDE.md                  # working agreement (section 9 of this file)
  README.md
  RESEARCH_LOG.md            # dated findings, decisions, rejected ideas
  pyproject.toml             # uv-managed, Python 3.12
  configs/
    default.yaml             # all tunables; nothing hardcoded in code
  data/
    raw/                     # untouched vendor downloads, gitignored
    processed/               # parquet, gitignored, rebuilt by pipeline
  src/usdjpy_mr/
    data/                    # loaders, alignment, validation
    stats/                   # stationarity, cointegration, OU, breaks
    backtest/                # vectorized + event-driven engines, costs
    signals/                 # live signal computation and schema
    export/                  # ibkr/, mt5/
    utils/
  notebooks/                 # exploratory only; nothing production depends on them
  tests/
  scripts/                   # CLI entry points: fetch, validate, test_stats, backtest, signal, export
  reports/                   # generated figures and tables, gitignored except summaries
```

## 4. Phase 1: Data pipeline

Sources (verify availability yourself; do not assume):

- US Treasury constant-maturity yields: FRED `DGS2`, `DGS10` (and `DFII10` for real yields). Use `fredapi` or direct CSV.
- JGB yields: Japan Ministry of Finance publishes daily historical JGB yields by tenor as CSV. Use 2y and 10y. Fallback: BoJ, or IBKR historical data for JGB futures if cash yields are unreliable.
- USD/JPY: start with `yfinance` (`USDJPY=X`) for research; add an IBKR loader (`ib_async`, spot `USD.JPY` on IDEALPRO, and `6J`/`JPY` futures on CME) as the production source. Note: IBKR requires a live TWS/Gateway session and market data permissions; build the loader but do not block Phase 1 on it.
- Optional later: Japan CPI and US CPI for real-yield variants.

Requirements:

- Timestamp alignment is the hard problem. JGB yields close in Tokyo, UST in New York, FX trades 24h. Choose one daily snapshot convention (default: New York 5pm close for FX and UST, previous Tokyo close for JGB, and document the lag). Make the convention a config option and record it in every processed file's metadata.
- Handle holidays for both calendars explicitly (Japan and US differ). Never forward-fill across a missing day silently; log it.
- Output: one tidy parquet `processed/daily.parquet` with columns `date, usdjpy, us2y, us10y, jgb2y, jgb10y, spread2y, spread10y, [us10y_real]`, plus a `metadata.json` with source URLs, download timestamps, alignment convention, and row counts.
- Validation script: no NaNs after alignment, no duplicate dates, yields within plausible bounds, FX within plausible bounds, gaps reported.
- Tests: loaders return expected schema, alignment logic is tested against a hand-built fixture, validation catches injected errors.

Acceptance: `scripts/fetch.py && scripts/validate.py` runs clean from an empty `data/` directory and produces at least 2010 to present.

## 5. Phase 2: Statistical confirmation

Run for each specification: spread tenor in {2y, 10y}, yield type in {nominal, real}, sample in {full, post-2016, post-2022}.

1. Stationarity of levels: ADF and KPSS on `usdjpy` and each spread. Expect I(1).
2. Cointegration: Engle-Granger (both orderings) and Johansen (trace and max-eigen). Report test statistics, p-values, and the cointegrating vector.
3. Beta estimation: static OLS, rolling OLS (windows 250, 500, 750), and a Kalman filter with time-varying beta. Plot beta paths; the Kalman path is the candidate for production if rolling OLS is unstable.
4. Residual dynamics: fit an OU process to the residual, report half-life with confidence interval (bootstrap). Report it per specification and per subsample.
5. Structural breaks: test at the known policy dates listed in section 1 (Chow) and search for unknown breaks (Bai-Perron or equivalent). Report whether beta or the residual variance breaks.
6. Rolling cointegration: recompute Engle-Granger on a rolling 500-day window; produce a time series of the test statistic so we can see when the relationship holds and when it does not.
7. Sanity check: the fitted residual must be strongly negative in Jan 2023 and strongly positive in Jul 2024. If it is not, something is wrong with alignment or sign conventions; stop and investigate.

Deliverable: `reports/phase2_summary.md` with a comparison table across specifications and a recommendation of one primary and one secondary specification. Write the reasoning into `RESEARCH_LOG.md`.

Acceptance: at least one specification shows cointegration at 5% on the post-2016 sample AND a half-life between 5 and 50 trading days AND no unexplained break in the last 24 months. If none does, say so plainly; that is a valid and important result.

## 6. Phase 3: Backtest

Two engines, both reading the same config:

- Vectorized (pandas/numpy) for parameter sweeps.
- Event-driven (daily loop, explicit position state) for the final numbers and as the reference implementation for the live signal engine. Both must agree on the base case to within rounding.

Signal logic (all thresholds from config, defaults shown):

- z-score of residual on rolling window (default 60 bars, also test 120 and half-life-based windows).
- Entry at |z| >= 2.0, exit at |z| <= 0.25, hard stop at |z| >= 3.5, time stop at 2x half-life.
- Sign: z < 0 means FX below fair value, long USD/JPY (and short the spread if two-leg); z > 0 the reverse.
- Position sizing: fixed notional first; volatility-scaled as a variant.

Costs, all mandatory, all configurable:

- FX spread and commission (use IBKR IDEALPRO-like assumptions; document them).
- Carry: the daily interest differential on the FX position. Short USD/JPY pays carry; this materially affects the z > 0 side. Compute it from the 2y or overnight differential in the data, not a constant.
- Bond leg: futures commissions and roll costs if two-leg.
- Slippage: a simple bps assumption, stressed higher in intervention periods.

Validation:

- Strict no look-ahead: beta, z-score window statistics, and half-life at time t use data up to t-1 only. Write a test that shifts the input and asserts signals do not change before the shift.
- Walk-forward: re-estimate beta and half-life on an expanding or rolling window, trade the next block, roll forward. Report in-sample vs out-of-sample separately; never report only in-sample.
- Parameter sweep over entry/exit/window; show the P&L surface is smooth, not a single spike.
- Robustness: results with and without the intervention windows; results per subsample; bootstrap Sharpe distribution and a deflated Sharpe ratio given the number of specifications tried.
- Metrics: CAGR, Sharpe, Sortino, max drawdown, hit rate, average holding period, turnover, profit factor, exposure, carry P&L broken out from spot P&L.

Deliverable: `reports/phase3_backtest.md` with tables and figures (equity curve, drawdown, z-score with entries/exits overlaid on the two historical anchors).

Acceptance: out-of-sample Sharpe > 0 after costs on the primary specification, drawdown explained by identifiable events, and no dependence on a single parameter cell. Otherwise stop and report.

## 7. Phase 4: Signal engine

- `scripts/signal.py` runs once per day after the chosen snapshot time, fetches the latest data increment, appends to the processed store, recomputes beta (per the chosen method), residual, z-score, half-life, and the discrete state.
- Output schema, written to `signals/latest.json` and appended to `signals/history.parquet`:

```json
{
  "schema_version": "1.0",
  "model_version": "<git sha of the config + code>",
  "as_of": "2026-09-04T21:00:00Z",
  "instrument": "USDJPY",
  "spec": {"tenor": "10y", "yield_type": "nominal", "beta_method": "kalman"},
  "inputs": {"usdjpy": 0.0, "us_yield": 0.0, "jgb_yield": 0.0, "spread": 0.0},
  "beta": 0.0, "alpha": 0.0,
  "fair_value": 0.0, "residual": 0.0, "zscore": 0.0, "half_life_days": 0.0,
  "state": "FLAT | LONG_USDJPY | SHORT_USDJPY",
  "action": "NONE | ENTER_LONG | ENTER_SHORT | EXIT | STOP",
  "hedge": {"enabled": false, "leg": "UST_vs_JGB_futures", "ratio": 0.0},
  "data_quality": {"stale_inputs": [], "warnings": []}
}
```

- The engine must refuse to emit an actionable signal if any input is stale beyond a configured tolerance or if the rolling cointegration test has failed for N consecutive windows (regime guard). It emits `state: FLAT, action: NONE` with the reason in `warnings`.
- The event-driven backtest engine and the live engine must share the same signal function. Test that replaying history through the live engine reproduces the backtest trades.

## 8. Phase 5: Export to IBKR TWS and MetaTrader 5

Design as a thin layer over `signals/latest.json`. The signal engine never talks to a broker directly.

IBKR (TWS / IB Gateway, `ib_async`):

- Mode A (default, paper account only): on `action != NONE`, place the corresponding order on `USD.JPY` IDEALPRO (and the futures hedge if enabled) with a configurable notional, log the order id, and reconcile positions on the next run. Always check current positions before acting; the signal is a target state, not a blind order.
- Mode B: create TWS price alerts / a small local dashboard (FastAPI + one page) that shows the current z-score and state, for manual execution. TWS has no clean custom-indicator mechanism, so this is the "indicator" surface on the IBKR side.
- Never run against the live account until Andrea explicitly changes a config flag named `ibkr.live_account_confirmed: true` AND the phase 3 acceptance criteria are met. Default to paper.

MetaTrader 5:

- Data/orders: the `MetaTrader5` Python package (Windows only, needs a running terminal). Build the bridge so it can run on a separate Windows machine reading `latest.json` over the network or a shared folder.
- Indicator: write an MQL5 custom indicator that displays z-score, fair value, and state as a subwindow on the USDJPY chart. Feed it either by having the Python side write a CSV into `MQL5/Files/` (simplest, use `FileOpen` with `FILE_COMMON`) or by the indicator polling a local HTTP endpoint via `WebRequest` (requires the URL to be whitelisted in the terminal). Implement the file approach first, HTTP second.
- MT5 broker quotes for USDJPY differ from IBKR spot. Compute the residual from the model's own inputs, not from the broker feed, and display the broker price separately.

Acceptance: end-to-end dry run where a synthetic signal file produces a paper order in TWS and a visible indicator value in MT5, with logs on both sides.

## 9. Working agreement (copy into CLAUDE.md)

- Python 3.12, `uv` for environment and lockfile, `ruff` for lint/format, `pytest` for tests, `pre-commit` configured. Core libs: `pandas`, `numpy`, `statsmodels`, `scipy`, `pyarrow`, `pydantic` for config, `fredapi`, `yfinance`, `ib_async`, `MetaTrader5` (optional extra, Windows only). Add anything else only with a one-line justification in the commit message.
- Everything tunable lives in `configs/*.yaml` and is validated by a pydantic model. No magic numbers in code.
- One branch per phase (`phase-1-data`, `phase-2-stats`, ...), conventional commits, open a PR at the end of each phase with a summary that mirrors the phase's acceptance criteria. Do not merge; Andrea merges.
- Tests must pass before every commit. New logic gets a test. Statistical functions get a test against a known synthetic series (e.g. a simulated OU process with known half-life).
- `RESEARCH_LOG.md`: dated entries. Record what was tried, what the numbers were, what was decided and why, and what was rejected. Rejected ideas are as important as accepted ones.
- Never fabricate data, test results, or numbers. If a data source is unavailable, say so and propose an alternative. If a test is inconclusive, report it as inconclusive.
- Secrets (FRED key, IBKR account, MT5 login) only via `.env`, which is gitignored. Add `.env.example`.
- Stop and ask before: changing the alignment convention after Phase 1, changing acceptance criteria, adding a data vendor that costs money, touching anything with `live` in its name.
- Keep responses to Andrea short: what was done, what the numbers say, what decision is needed.

## 10. Open decisions (ask Andrea before Phase 1 ends)

- [ ] FRED API key available? (`.env`)
- [ ] Is TWS/IB Gateway running on the dev machine, and are FX and CME futures market data subscriptions active?
- [ ] Which MT5 broker/account, and is there a Windows machine for the MT5 bridge?
- [ ] Preferred default snapshot convention (NY 5pm vs Tokyo close) if the analysis in Phase 1 does not make one clearly better.
- [ ] Two-leg hedge: is trading JGB futures on OSE actually enabled on the IBKR account, or is the hedge research-only for now?
