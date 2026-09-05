# Working agreement

Project: USD/JPY vs US–Japan yield spread, mean-reversion signal engine.
Full brief: `docs/HANDOFF.md`. Read it end to end before writing code.

- Python 3.12, `uv` for environment and lockfile, `ruff` for lint/format, `pytest` for tests, `pre-commit` configured. Core libs: `pandas`, `numpy`, `statsmodels`, `scipy`, `pyarrow`, `pydantic` for config, `fredapi`, `yfinance`, `ib_async`, `MetaTrader5` (optional extra, Windows only). Add anything else only with a one-line justification in the commit message.
- Everything tunable lives in `configs/*.yaml` and is validated by a pydantic model. No magic numbers in code.
- One branch per phase (`phase-1-data`, `phase-2-stats`, ...), conventional commits, open a PR at the end of each phase with a summary that mirrors the phase's acceptance criteria. Andrea reads the summary and gives the go; Claude merges on that go, never before it (agreed 2026-09-05).
- Tests must pass before every commit. New logic gets a test. Statistical functions get a test against a known synthetic series (e.g. a simulated OU process with known half-life).
- `RESEARCH_LOG.md`: dated entries. Record what was tried, what the numbers were, what was decided and why, and what was rejected. Rejected ideas are as important as accepted ones.
- Never fabricate data, test results, or numbers. If a data source is unavailable, say so and propose an alternative. If a test is inconclusive, report it as inconclusive.
- Secrets (FRED key, IBKR account, MT5 login) only via `.env`, which is gitignored. Add `.env.example`.
- Stop and ask before: changing the alignment convention after Phase 1, changing acceptance criteria, adding a data vendor that costs money, touching anything with `live` in its name.
- Keep responses to Andrea short: what was done, what the numbers say, what decision is needed.

## Phase gating

Phases run strictly in order (data → stats → backtest → signal engine → export). Each phase ends with a commit, a `RESEARCH_LOG.md` entry, and a stop for review. If a phase fails its acceptance criteria, report that and stop; never loosen the criteria to pass.
