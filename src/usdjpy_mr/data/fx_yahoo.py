"""USD/JPY daily closes from Yahoo Finance via yfinance (research source).

Yahoo labels its daily FX bar with the date on which the bar ends (about 00:00 GMT), so the
row labelled D actually holds the NY 17:00 ET close of the previous weekday. Verified against
IBKR IDEALPRO daily closes on 2026-08-28..2026-09-03: Yahoo[2026-08-31] = IBKR close of
2026-08-28, Yahoo[2026-09-03] = IBKR close of 2026-09-02. `relabel_by_weekdays` moves every row
back by `yahoo_label_offset_bdays` (config, default -1) so that the value on D is the NY close
of D. IBKR is the production source (see ibkr.py).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from pathlib import Path

import numpy as np
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


def relabel_by_weekdays(s: pd.Series, offset_bdays: int) -> pd.Series:
    """Move each observation's date by `offset_bdays` weekdays (Mon-Fri), keeping values.
    A weekend label is first rolled back to the preceding weekday. Duplicate targets keep the
    last observation."""
    if offset_bdays == 0:
        return s
    days = s.index.to_numpy().astype("datetime64[D]")
    moved = np.busday_offset(days, offset_bdays, roll="backward")
    out = pd.Series(s.to_numpy(), index=pd.DatetimeIndex(moved, name="date"), name=s.name)
    return out[~out.index.duplicated(keep="last")].sort_index()


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
    s = relabel_by_weekdays(s, cfg.yahoo_label_offset_bdays)
    if cfg.yahoo_label_offset_bdays:
        log.info("Yahoo rows relabelled by %+d weekday(s)", cfg.yahoo_label_offset_bdays)
    log.info(
        "Yahoo %s: %d rows, %s -> %s",
        cfg.yahoo_ticker,
        len(s),
        s.index.min().date(),
        s.index.max().date(),
    )
    return s, rec
