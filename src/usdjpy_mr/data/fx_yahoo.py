"""USD/JPY daily closes from Yahoo Finance via yfinance (research source).

Yahoo's daily FX bar for date D closes around 21:00-23:00 UTC, i.e. near the NY 17:00 ET
FX day roll, so `Close` on D is treated as the NY-close snapshot of D. IBKR is the production
source (see ibkr.py); differences between the two are reported in Phase 1 validation.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from usdjpy_mr.config import FxConfig
from usdjpy_mr.utils.io import RawFile, sha256_bytes, utc_now_iso
from usdjpy_mr.utils.logging import get_logger

log = get_logger(__name__)

Downloader = Callable[[str, dt.date, dt.date | None], pd.DataFrame]


def _yf_download(ticker: str, start: dt.date, end: dt.date | None) -> pd.DataFrame:
    import yfinance as yf  # lazy import: not needed in tests

    # end is exclusive in yfinance; add one day so `end` itself is included.
    end_excl = (end + dt.timedelta(days=1)).isoformat() if end else None
    return yf.download(
        ticker,
        start=start.isoformat(),
        end=end_excl,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )


def normalise_yahoo_frame(df: pd.DataFrame, field: str, ticker: str) -> pd.Series:
    """Flatten yfinance output (single- or multi-index columns) to one float Series."""
    if df is None or df.empty:
        raise ValueError(f"yfinance returned no rows for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        if field not in lvl0:
            raise ValueError(f"field {field!r} not in yfinance columns {list(lvl0.unique())}")
        sub = df[field]
        s = sub[ticker] if ticker in sub.columns else sub.iloc[:, 0]
    else:
        if field not in df.columns:
            raise ValueError(f"field {field!r} not in yfinance columns {list(df.columns)}")
        s = df[field]
    s = pd.Series(
        pd.to_numeric(s, errors="coerce").to_numpy(),
        index=pd.to_datetime(df.index),
        name="usdjpy",
        dtype="float64",
    )
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    s.index = s.index.normalize()
    s.index.name = "date"
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.dropna()


def fetch_fx_yahoo(
    cfg: FxConfig,
    start: dt.date,
    end: dt.date | None,
    raw_dir: Path,
    downloader: Downloader = _yf_download,
) -> tuple[pd.Series, RawFile]:
    log.info("Yahoo %s from %s", cfg.yahoo_ticker, start)
    raw = downloader(cfg.yahoo_ticker, start, end)
    dest = raw_dir / "yahoo" / f"{cfg.yahoo_ticker.replace('=', '_')}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    body = raw.to_csv().encode()
    dest.write_bytes(body)
    rec = RawFile(
        source="yahoo:" + cfg.yahoo_ticker,
        url=f"yfinance://{cfg.yahoo_ticker}",
        path=str(dest),
        downloaded_at=utc_now_iso(),
        sha256=sha256_bytes(body),
        bytes=len(body),
    )
    s = normalise_yahoo_frame(raw, cfg.yahoo_field, cfg.yahoo_ticker)
    log.info(
        "Yahoo %s: %d rows, %s -> %s",
        cfg.yahoo_ticker,
        len(s),
        s.index.min().date(),
        s.index.max().date(),
    )
    return s, rec
