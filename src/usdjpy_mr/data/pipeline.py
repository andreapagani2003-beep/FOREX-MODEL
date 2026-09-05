"""Fetch -> align -> write processed/daily.parquet + metadata.json."""

from __future__ import annotations

import datetime as dt
import json
import platform
from pathlib import Path

import pandas as pd

from usdjpy_mr import __version__
from usdjpy_mr.config import Config
from usdjpy_mr.data.align import AlignmentReport, align_daily
from usdjpy_mr.data.fred import fetch_fred_all
from usdjpy_mr.data.fx_yahoo import fetch_fx_yahoo
from usdjpy_mr.data.mof import fetch_mof
from usdjpy_mr.utils.io import RawFile, git_sha, utc_now_iso
from usdjpy_mr.utils.logging import get_logger

log = get_logger(__name__)


def fetch_all(
    cfg: Config, fx_provider: str | None = None
) -> tuple[dict[str, pd.Series], list[RawFile]]:
    """Download every source into data/raw and return {column: Series} plus provenance."""
    start = cfg.project.start_date
    end = cfg.project.effective_end_date()
    # Fetch a little earlier than start so as-of lookups at the start of the window can resolve.
    fetch_start = start - dt.timedelta(days=31)
    raw_dir = cfg.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    fred_df, fred_recs = fetch_fred_all(cfg.sources.fred, fetch_start, raw_dir)
    mof_df, mof_recs = fetch_mof(cfg.sources.mof, fetch_start, raw_dir)
    provider = fx_provider or cfg.sources.fx.provider
    if provider == "yahoo":
        fx, fx_rec = fetch_fx_yahoo(cfg.sources.fx, fetch_start, end, raw_dir)
    elif provider == "ibkr":
        from usdjpy_mr.data.ibkr import fetch_fx_ibkr

        fx, fx_rec = fetch_fx_ibkr(cfg.sources.ibkr, fetch_start, end, raw_dir)
    else:
        raise ValueError(f"unknown fx provider {provider!r}")

    sources: dict[str, pd.Series] = {"usdjpy": fx}
    for col in fred_df.columns:
        sources[col] = fred_df[col]
    for col in mof_df.columns:
        sources[col] = mof_df[col]
    return sources, [*fred_recs, *mof_recs, fx_rec]


def build_daily(
    cfg: Config, sources: dict[str, pd.Series], raw_files: list[RawFile]
) -> tuple[pd.DataFrame, dict]:
    """Align sources and write daily.parquet + metadata.json. Returns (frame, metadata)."""
    start = cfg.project.start_date
    end = cfg.project.effective_end_date()
    df, report = align_daily(sources, cfg.alignment, start, end)
    metadata = _metadata(cfg, sources, raw_files, df, report)
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cfg.daily_parquet, engine="pyarrow", index=True)
    cfg.metadata_json.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    log.info("wrote %s (%d rows) and %s", cfg.daily_parquet, len(df), cfg.metadata_json)
    return df, metadata


def _metadata(
    cfg: Config,
    sources: dict[str, pd.Series],
    raw_files: list[RawFile],
    df: pd.DataFrame,
    report: AlignmentReport,
) -> dict:
    return {
        "schema_version": "1.0",
        "package_version": __version__,
        "git_sha": git_sha(cfg.root),
        "built_at": utc_now_iso(),
        "python": platform.python_version(),
        "project": {
            "start_date": str(cfg.project.start_date),
            "end_date": str(cfg.project.effective_end_date()),
        },
        "row_count": len(df),
        "first_date": str(df.index.min().date()) if len(df) else None,
        "last_date": str(df.index.max().date()) if len(df) else None,
        "columns": list(df.columns),
        "sources": [r.as_dict() for r in raw_files],
        "source_notes": {
            "fred": cfg.sources.fred.snapshot_note,
            "mof": cfg.sources.mof.snapshot_note,
            "fx": cfg.sources.fx.snapshot_note,
        },
        "source_rows": {k: int(v.dropna().shape[0]) for k, v in sources.items()},
        "source_ranges": {
            k: [str(v.dropna().index.min().date()), str(v.dropna().index.max().date())]
            for k, v in sources.items()
            if v.dropna().shape[0]
        },
        "alignment": report.as_dict(),
        "dropped_dates": _dropped_summary(report),
        "config": cfg.model_dump(mode="json"),
    }


def _dropped_summary(report: AlignmentReport) -> list[dict]:
    if not len(report.dropped):
        return []
    g = report.dropped.groupby("date").agg(columns=("column", list), reasons=("reason", "unique"))
    return [
        {"date": d.strftime("%Y-%m-%d"), "columns": c, "reasons": list(r)}
        for d, c, r in zip(g.index, g["columns"], g["reasons"], strict=True)
    ]


def load_daily(cfg: Config) -> tuple[pd.DataFrame, dict]:
    df = pd.read_parquet(cfg.daily_parquet, engine="pyarrow")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    metadata = json.loads(Path(cfg.metadata_json).read_text(encoding="utf-8"))
    return df, metadata
