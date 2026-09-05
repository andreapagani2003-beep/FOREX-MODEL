"""US Treasury constant-maturity yields from FRED.

Two paths, same output: `fredapi` when an API key is present, otherwise the keyless
`fredgraph.csv` endpoint. Both produce a float Series indexed by naive date, NaN on holidays.
"""

from __future__ import annotations

import datetime as dt
import io
from pathlib import Path

import pandas as pd

from usdjpy_mr.config import FredConfig
from usdjpy_mr.utils.io import RawFile, download_raw, sha256_bytes, utc_now_iso
from usdjpy_mr.utils.logging import get_logger

log = get_logger(__name__)


def parse_fred_csv(text: str, series_id: str) -> pd.Series:
    """Parse a fredgraph.csv body: header `DATE,<id>` or `observation_date,<id>`; '.' = NaN."""
    df = pd.read_csv(io.StringIO(text), na_values=["."], dtype=str)
    if df.shape[1] < 2:
        raise ValueError(f"FRED CSV for {series_id}: expected 2 columns, got {list(df.columns)}")
    date_col, value_col = df.columns[0], df.columns[1]
    if value_col.strip().upper() != series_id.upper():
        log.warning("FRED CSV value column %r != requested %r", value_col, series_id)
    out = pd.Series(
        pd.to_numeric(df[value_col].str.strip(), errors="coerce").to_numpy(),
        index=pd.to_datetime(df[date_col].str.strip(), format="%Y-%m-%d"),
        name=series_id,
        dtype="float64",
    )
    out.index.name = "date"
    return out.sort_index()


def _via_api(cfg: FredConfig, series_id: str, start: dt.date, api_key: str) -> pd.Series:
    from fredapi import Fred  # lazy: optional runtime path

    fred = Fred(api_key=api_key)
    s = fred.get_series(series_id, observation_start=start.isoformat())
    s = pd.Series(pd.to_numeric(s, errors="coerce"), name=series_id, dtype="float64")
    s.index = pd.to_datetime(s.index)
    s.index.name = "date"
    return s.sort_index()


def fetch_fred_series(
    cfg: FredConfig, series_id: str, start: dt.date, raw_dir: Path
) -> tuple[pd.Series, RawFile]:
    """Fetch one series from `start`, archive the raw payload, return (series, provenance)."""
    api_key = cfg.api_key()
    if api_key:
        log.info("FRED %s via fredapi (key from $%s)", series_id, cfg.api_key_env)
        s = _via_api(cfg, series_id, start, api_key)
        dest = raw_dir / "fred" / f"{series_id}.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        body = s.to_csv(header=True).encode()
        dest.write_bytes(body)
        rec = RawFile(
            source=f"fred:{series_id}",
            url=f"fredapi://series/{series_id}",
            path=str(dest),
            downloaded_at=utc_now_iso(),
            sha256=sha256_bytes(body),
            bytes=len(body),
        )
    else:
        url = cfg.csv_url_template.format(series=series_id)
        log.info("FRED %s via keyless CSV (no $%s set)", series_id, cfg.api_key_env)
        data, rec = download_raw(f"fred:{series_id}", url, raw_dir / "fred" / f"{series_id}.csv")
        s = parse_fred_csv(data.decode("utf-8"), series_id)
    s = s.loc[s.index >= pd.Timestamp(start)]
    log.info(
        "FRED %s: %d rows, %s -> %s, %d NaN",
        series_id,
        len(s),
        s.index.min().date(),
        s.index.max().date(),
        int(s.isna().sum()),
    )
    return s, rec


def fetch_fred_all(
    cfg: FredConfig, start: dt.date, raw_dir: Path
) -> tuple[pd.DataFrame, list[RawFile]]:
    """All configured series as one wide frame (columns = config keys, e.g. us2y, us10y)."""
    cols: dict[str, pd.Series] = {}
    recs: list[RawFile] = []
    for name, series_id in cfg.series.items():
        s, rec = fetch_fred_series(cfg, series_id, start, raw_dir)
        cols[name] = s
        recs.append(rec)
    df = pd.DataFrame(cols)
    df.index.name = "date"
    return df.sort_index(), recs
