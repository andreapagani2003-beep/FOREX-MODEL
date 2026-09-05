# Research log

Dated entries: what was tried, the numbers, what was decided and why, what was rejected.

## 2026-09-05 — Project setup and Phase 1 (data pipeline) build

**Context.** Handoff received (`docs/HANDOFF.md`). Working agreement copied to `CLAUDE.md`.
Andrea's answers to the open decisions: FRED key to be created (free); IBKR paper account only
until the model is finished; development on a Windows machine; MT5 is a later favour for a
friend. Checked the IBKR account through the IBKR connector: USD.JPY on IDEALPRO returns quotes,
UST futures (ZN, CBOT) and JGB futures (OSE) return no market data (no futures permissions or
subscriptions), so the two-leg hedge is research-only for now. The connected account shows a EUR
base and ~€215 net liquidation, i.e. it looks like the live account; nothing here touches it.

**Built.** `uv` project (Python 3.12), pydantic-validated config (`configs/default.yaml`,
`configs/tokyo_close.yaml`), loaders for FRED (fredapi or keyless CSV), MoF JGB CSV (header
auto-detected, encodings tried in order, `-` → NaN), Yahoo FX (yfinance, multi-index safe), IBKR
FX (ib_async, lazy import, opt-in via `--fx-provider ibkr`), holiday calendars (`holidays`
package: NYSE financial calendar for the US, JP public holidays plus Dec 31 / Jan 2–3 market
closures), alignment with explicit lag columns, validation, pipeline, `scripts/fetch.py`,
`scripts/validate.py`, and `notebooks/phase1_data.ipynb` (Colab + VS Code). 43 unit tests, all
passing; ruff clean.

**Decisions.**
- Default snapshot convention `ny_close`: FX and UST are NY-close quantities and are taken on
  date t; the JGB print of t (15:00 JST) precedes NY 17:00 by ~11h, so it is also taken on t.
  This gives zero lag on all three legs on a normal day, which is why it is the default. The
  `tokyo_close` mirror (JGB same-day, FX/UST from t-1) is implemented for comparison in Phase 2.
  Changing the convention after Phase 1 requires a stop-and-ask per the working agreement.
- Missing-data policy: FX and UST must be observed on the snapshot date, otherwise the row is
  dropped (US holidays: no UST print, no new spread information). JGB is carried forward at most
  7 calendar days (`asof_max_lag_days`), with the lag stored in `jgb_lag_days`. 7 was chosen so
  that ordinary Japan holidays and year-end (≤ 6 weekdays) never drop a row, while a 10-day
  staleness (2019 Golden Week, 27 Apr–6 May) drops exactly one row (2019-05-06). Rejected:
  unlimited forward-fill (hides staleness) and dropping every Japan holiday (loses ~16 rows a year
  for no reason, since the snapshot is NY-centric).
- Extra columns `fx_lag_days`, `us_lag_days`, `jgb_lag_days` are kept in `daily.parquet` beyond
  the handoff schema so staleness is visible downstream (Phase 4 regime guard will use them).
- Added dependencies beyond the core list: `holidays` (classify gaps as holiday vs unexplained),
  `pyyaml` (config), `python-dotenv` (`.env`). Justified in the commit message.

**Numbers (synthetic dry run, holiday-aware, 2010–2024).** 3,913 weekdays → 3,773 rows; 139
dates dropped for US holidays (no UST print), 1 for JGB staleness (2019-05-06); 225 rows carry a
JGB value forward (Japan holidays), mean JGB lag 0.15 days, max 7; zero unexplained gaps. These
are pipeline-behaviour numbers on synthetic inputs, not market data.

**Not done / blocked.** The development container's egress policy blocks fred.stlouisfed.org,
mof.go.jp and finance.yahoo.com, so the real fetch could not be run here. The MoF and FRED parsers
were written against the documented formats and tested on hand-written fixtures; the real files
may differ (title lines, encoding, column names) and the parsers are defensive about that, but
this is unverified. **Phase 1 acceptance (fetch + validate clean from empty `data/`, 2010 →
present) is pending Andrea's run of `notebooks/phase1_data.ipynb` on Colab.** No PR until then.

## 2026-09-05 — Phase 1 acceptance run (Colab, real data)

**Run.** `notebooks/phase1_data.ipynb` on Colab with a FRED key, `ny_close` convention, from an
empty `data/`. Fetch, align and validate all completed; validation **PASS**, 0 errors, 0 warnings.

| metric | value |
|---|---|
| rows | 4,164 |
| range | 2010-01-04 → 2026-09-03 |
| weekdays in range | 4,349 (coverage 95.75%) |
| dates dropped | 187: 180 no UST print (US bond holidays), 10 no Yahoo FX bar, 1 JGB stale > 7 days |
| JGB carried forward | 239 rows, mean lag 0.14 days, max 7 |
| largest gap | 2 weekdays |
| ranges | USD/JPY 75.7–163.9; UST 2y 0.09–5.19; UST 10y 0.52–4.98; JGB 2y −0.37–1.85; JGB 10y −0.30–3.01 |
| spreads | 2y 0.03–5.13; 10y 0.50–4.14 |

**Anchors (handoff §1).** Jan 2023: USD/JPY 128.3–132.1 with 10y spread 2.92–3.10. Jul 2024:
USD/JPY 158.2–161.6 with 10y spread 3.11–3.22. Both consistent with the handoff's approximate
values (128 / 3.1 and 161 / 3.4); the Jul 2024 spread is ~0.2pp lower than the handoff figure,
which is within what different tenor/quote conventions produce. The sign test on the fitted
residual is Phase 2 work.

**Source observations.**
- FRED via `fredapi`: 4,373 rows per series, 180 NaN each (holidays), last print 2026-09-03.
- MoF: the all-history file runs to 2026-09-03 (1.19 MB); the current-year file is tiny (545 B).
  A footer line ("※If you cannot download the latest csv data...") is dropped by the parser and
  logged, as designed. No NaN in 2Y/10Y from 2010.
- Yahoo `USDJPY=X`: 4,364 rows, last bar 2026-09-04. Six weekday bars missing that are not
  holidays (2011-04-15, 2013-10-08, 2017-07-11, 2017-11-16, 2019-05-22, 2025-04-21): Yahoo data
  holes, rows dropped and logged. IBKR as production source removes this.
- 31 UST gaps were reported "unexplained": all Columbus Day and Veterans Day, when the bond market
  closes but the NYSE does not. **Fixed**: US calendar is now NYSE ∪ US federal holidays, i.e.
  the SIFMA full-close list. The single JGB gap (2026-09-04) is publication lag; gaps within
  `alignment.publication_lag_days` (3) of the end date are now classified as such.

**Decision.** Phase 1 acceptance criteria met (fetch + validate clean from empty `data/`, 2010 →
present). Convention stays `ny_close`. PR opened for review; Phase 2 starts on approval.

## 2026-09-05 — Phase 2: statistical confirmation. Acceptance NOT met.

**Run.** `scripts/test_stats.py` on the Phase 1 parquet (4,164 rows, 2010-01-04 → 2026-09-03,
`ny_close`). 3 specs (10y nominal, 2y nominal, 10y US-real minus nominal JGB) × 3 samples
(full, post-2016, post-2022). Full table and figures: `reports/phase2_summary.md`,
`reports/figures/phase2_*.png`, raw numbers in `reports/phase2/results.json`.

**What the numbers say.**

| spec / sample | EG p (spot on spread) | Johansen trace r=0 (crit95 15.5) | beta (static) | OU half-life, days (95% CI) |
|---|---|---|---|---|
| 10y nominal / full | 0.97 | 4.9 | 19.0 | 359 (116–643) |
| 10y nominal / post-2016 | 0.99 | 2.8 | 13.9 | 648 (90–∞) |
| 10y nominal / post-2022 | 0.36 | 12.0 | 0.8 (R² 0.002) | 122 (51–219) |
| 2y nominal / full | 0.96 | 1.6 | 12.7 | 438 (116–819) |
| 2y nominal / post-2016 | 0.99 | 2.4 | 10.3 | 455 (82–∞) |
| 2y nominal / post-2022 | 0.52 | 18.0 (rank 1) | 1.6 (R² 0.02) | 135 (50–257) |
| 10y real / full | 0.99 | 7.0 | 18.0 | 619 (139–∞) |
| 10y real / post-2016 | 0.99 | 2.9 | 9.8 | 2101 (109–∞) |
| 10y real / post-2022 | 0.49 | 11.4 | 2.2 (R² 0.02) | 132 (50–249) |

- Levels are I(1) for spot and for every spread (ADF/KPSS agree), as expected.
- **No specification is cointegrated at 5% on the full or post-2016 sample**, by either test.
  Engle-Granger p-values are 0.95–0.99 in both orderings; Johansen trace statistics are 2–7
  against a critical value of 15.5.
- **Half-lives are 360–2100 days**, an order of magnitude outside the 5–50 band. The residual
  is a slow random walk, not a mean-reverting spread.
- Rolling 500-day Engle-Granger rejects in only 7–11% of windows, essentially all in 2020
  (window-end median p 0.01 in 2020, 0.2–0.9 in every other year). The relationship held for one
  stretch and nowhere else.
- Beta is unstable: rolling-500 beta ranges −34 to +24 (10y), and sequential sup-F finds sign
  flips after Aug 2023 / Jan 2024 (beta +15 → −4) and again in Mar 2025 (+9 → −13). Post-2022
  the level regression has R² ≈ 0 and beta ≈ 0. Chow tests at the seven known policy dates all
  show enormous classical F, but sieve-bootstrap p-values of 0.06–0.09: with a residual this
  persistent, even a huge coefficient change is not distinguishable from drift, which is the same
  conclusion as the cointegration tests seen from the other side.
- The yearly picture makes it concrete: by the full-sample fit, USD/JPY was 25 below fair value
  in 2010–11, 18 below in 2018, 15 above in 2020, and 48 above in 2026 (spot 159 with a 10y
  spread of 1.9pp, versus 128 with 3.0pp in Jan 2023). The long-run level mapping has drifted
  far more than it has reverted.
- Sanity anchors: static residual is negative in Jan 2023 (−2.3, z −0.5) and positive in Jul
  2024 (+25, z +1.6). Signs match the handoff, but Jan 2023 is only mildly negative, not
  "strongly"; this is a weak pass and does not rescue the acceptance criteria.
- The only 5% rejection anywhere is Johansen rank 1 for 2y nominal post-2022 (trace 18.0 vs
  15.5) with EG p 0.52 and a 135-day half-life: not confirmed by the second test and far outside
  the band.

**Diagnostics tried and rejected (not acceptance, documented so they are not re-tried).**
log(spot) instead of level: EG p 0.95 (full) / 0.99 (post-2016). Adding a deterministic trend
(`ct`): 0.88 / 0.36 level, 0.82 / 0.19 log. Nothing approaches 5%, and a trend term would be a
fitted excuse for the drift, not evidence of equilibrium.

**Method notes.** Two things had to be done properly: (1) half-life uses the exact mapping
theta = −ln(1 + slope) rather than −slope, which biases short half-lives upward; (2) break-test
p-values use a sieve (AR(p)) bootstrap, because an iid residual bootstrap flagged spurious breaks
on synthetic OU residuals (a persistent residual makes any split look significant). The
classical Chow p-values are printed for reference only.

**Decision.** Per the working agreement, Phase 2 fails its acceptance criteria and the pipeline
stops here. Nothing was loosened. A static or slowly-varying linear level relationship between
USD/JPY and the US–JGB yield spread is not supported over 2010–2026; the spread explains the
direction of big moves (2022, 2024) but not a stable level to revert to. Options for Andrea to
decide, none of which is the current model: (a) a regime-conditional relationship (Markov
switching on the residual dynamics or the beta), (b) a richer fair-value model (add terms such as
risk sentiment, BoJ balance sheet, terms of trade) and re-test cointegration, (c) a shorter-horizon
error-correction specification in changes rather than levels, (d) accept that 2020-style episodes
are the only cointegrated regime and treat the strategy as episodic. Each is a new hypothesis
with its own Phase 2, not a tweak of this one.
