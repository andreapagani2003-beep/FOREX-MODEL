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
