# FOREX-MODEL — USD/JPY vs US–Japan yield spread

Cointegration mean-reversion signal engine: `USDJPY_t = alpha + beta * spread_t + eps_t`,
trade the residual via a rolling z-score. Full brief in `docs/HANDOFF.md`; working agreement in
`CLAUDE.md`; findings and decisions in `RESEARCH_LOG.md`.

## Status

| Phase | Branch | State |
|---|---|---|
| 1 Data pipeline | `phase-1-data` | acceptance met on real data (4,164 rows, 2010-01-04 → 2026-09-03); PR open |
| 2 Statistical confirmation | `phase-2-stats` | **acceptance NOT met**: no cointegration at 5% on full/post-2016, half-lives 360–2100 days; see `reports/phase2_summary.md` |
| 3 Backtest | | not started |
| 4 Signal engine | | not started |
| 5 Export (IBKR TWS, MT5) | | not started |

## Quick start (local / VS Code)

```bash
uv sync --group dev                 # Python 3.12 env + lockfile
cp .env.example .env                # add FRED_API_KEY (optional: keyless CSV fallback exists)
uv run scripts/fetch.py             # -> data/processed/daily.parquet + metadata.json
uv run scripts/validate.py          # -> reports/phase1_validation.md, exit 1 on any error
uv run pytest                       # unit tests (no network needed)
uv run scripts/test_stats.py        # Phase 2: reports/phase2_summary.md + figures (exit 2 if acceptance not met)
```

## Quick start (Google Colab)

Open `notebooks/phase1_data.ipynb` in Colab (File → Open notebook → GitHub → this repo, branch
`phase-1-data`). The first cell clones the repo and installs the package. Add `FRED_API_KEY` as a
Colab secret (key icon in the sidebar) if you have one.

## Layout

```
configs/        default.yaml (NY-close convention), tokyo_close.yaml (alternative)
src/usdjpy_mr/  data/ (loaders, calendars, align, validate, pipeline) · stats/ · backtest/ · signals/ · export/
scripts/        fetch.py, validate.py
tests/          fixtures + unit tests
notebooks/      exploratory only; nothing production depends on them
data/           raw/ (untouched vendor files) and processed/ (parquet) — gitignored
reports/        generated; gitignored except phase summaries
```

## Data conventions (Phase 1)

- Sources: FRED `DGS2`, `DGS10`, `DFII10`; Japan MoF JGB benchmark yields (2Y, 10Y); Yahoo
  `USDJPY=X` for research, IBKR IDEALPRO `USD.JPY` as production source (loader built, opt-in).
- Snapshot convention `ny_close` (default): NY 17:00 on date t; FX and UST are same-day, JGB is
  the Tokyo close of t (~11h earlier), or the previous Tokyo close on Japan holidays with the lag
  recorded in `jgb_lag_days`. `tokyo_close` is the mirror image (`configs/tokyo_close.yaml`).
- Rows without a same-day FX or UST observation are dropped and logged; JGB may be carried forward
  at most 7 calendar days; nothing is forward-filled silently. Every gap is classified as a known
  holiday or reported as unexplained in `metadata.json`.
