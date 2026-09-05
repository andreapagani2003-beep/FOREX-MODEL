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
