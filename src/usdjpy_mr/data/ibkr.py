"""USD/JPY history from Interactive Brokers via ib_async (production FX source).

Requires a running TWS / IB Gateway with API access enabled and FX market data permission.
Phase 1 does not depend on this loader; it is exercised only when explicitly requested
(`scripts/fetch.py --fx-provider ibkr`). Paper account by default (port 7497).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from usdjpy_mr.config import IbkrConfig
from usdjpy_mr.utils.io import RawFile, sha256_bytes, utc_now_iso
from usdjpy_mr.utils.logging import get_logger

log = get_logger(__name__)


def _duration_string(start: dt.date, end: dt.date) -> str:
    years = max(1, (end - start).days // 365 + 1)
    return f"{years} Y"


def bars_to_series(bars: list, name: str = "usdjpy") -> pd.Series:
    """Convert ib_async BarData list to a float Series indexed by naive date."""
    if not bars:
        raise ValueError("IBKR returned no bars")
    idx = pd.to_datetime([b.date for b in bars])
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    s = pd.Series([float(b.close) for b in bars], index=idx.normalize(), name=name, dtype="float64")
    s.index.name = "date"
    return s[~s.index.duplicated(keep="last")].sort_index()


def fetch_fx_ibkr(
    cfg: IbkrConfig, start: dt.date, end: dt.date, raw_dir: Path
) -> tuple[pd.Series, RawFile]:
    """Daily USD.JPY midpoint closes from IDEALPRO. Blocks until TWS answers."""
    from ib_async import IB, Forex  # lazy: only needed when a TWS session exists

    if cfg.live_account_confirmed:
        raise RuntimeError("ibkr.live_account_confirmed is true; refusing in Phase 1")
    ib = IB()
    host, port, client_id = cfg.host(), cfg.port(), cfg.client_id()
    log.info("IBKR connect %s:%d clientId=%d", host, port, client_id)
    ib.connect(host, port, clientId=client_id, readonly=True, timeout=20)
    try:
        contract = Forex(pair=f"{cfg.spot.symbol}{cfg.spot.currency}", exchange=cfg.spot.exchange)
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=f"{end:%Y%m%d} 23:59:59",
            durationStr=_duration_string(start, end),
            barSizeSetting=cfg.bar_size,
            whatToShow=cfg.spot.what_to_show,
            useRTH=cfg.use_rth,
            formatDate=1,
        )
    finally:
        ib.disconnect()
    s = bars_to_series(bars)
    s = s.loc[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]
    dest = raw_dir / "ibkr" / f"{cfg.spot.symbol}{cfg.spot.currency}_{cfg.spot.what_to_show}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    body = s.to_csv(header=True).encode()
    dest.write_bytes(body)
    rec = RawFile(
        source="ibkr:" + cfg.spot.symbol + cfg.spot.currency,
        url=f"ibkr://{host}:{port}/{cfg.spot.exchange}/{cfg.spot.symbol}.{cfg.spot.currency}",
        path=str(dest),
        downloaded_at=utc_now_iso(),
        sha256=sha256_bytes(body),
        bytes=len(body),
    )
    log.info("IBKR %s: %d rows", contract.pair(), len(s))
    return s, rec
