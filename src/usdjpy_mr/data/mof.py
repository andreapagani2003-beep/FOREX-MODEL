"""JGB benchmark yields from Japan's Ministry of Finance.

MoF publishes two English CSVs: the full history (`jgbcme_all.csv`) and the current year
(`jgbcme.csv`). Format observed historically: one or more title lines, then a header row
`Date,1Y,2Y,...,40Y`, dates as `YYYY/M/D`, missing values as `-`. The parser locates the
header row instead of assuming its position, and tolerates either encoding.
"""

from __future__ import annotations

import datetime as dt
import io
from pathlib import Path

import pandas as pd

from usdjpy_mr.config import MofConfig
from usdjpy_mr.utils.io import RawFile, download_raw
from usdjpy_mr.utils.logging import get_logger

log = get_logger(__name__)

_NA = ["-", "", "n/a", "N/A", "ND"]


def decode_bytes(data: bytes, candidates: list[str]) -> str:
    last: UnicodeDecodeError | None = None
    for enc in candidates:
        try:
            return data.decode(enc)
        except UnicodeDecodeError as exc:  # try next
            last = exc
    raise ValueError(f"could not decode MoF CSV with {candidates}: {last}")


def parse_mof_csv(text: str) -> pd.DataFrame:
    """Parse a MoF JGB yield CSV into a float frame indexed by date, columns like '2Y', '10Y'."""
    lines = text.splitlines()
    header_idx = next(
        (i for i, ln in enumerate(lines) if ln.strip().lstrip("﻿").lower().startswith("date")),
        None,
    )
    if header_idx is None:
        raise ValueError("MoF CSV: no header row starting with 'Date' found")
    body = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(body), dtype=str, na_values=_NA, keep_default_na=True)
    df.columns = [c.strip().lstrip("﻿").upper() for c in df.columns]
    if "DATE" not in df.columns:
        raise ValueError(f"MoF CSV: header row lacks 'Date': {list(df.columns)}")
    df = df.dropna(subset=["DATE"])
    dates = pd.to_datetime(df["DATE"].str.strip(), format="mixed", errors="coerce")
    bad = dates.isna()
    if bad.any():
        log.warning(
            "MoF CSV: dropping %d rows with unparseable dates (e.g. %r)",
            int(bad.sum()),
            df.loc[bad, "DATE"].iloc[0],
        )
    out = (
        df.loc[~bad]
        .drop(columns=["DATE"])
        .apply(lambda c: pd.to_numeric(c.str.strip(), errors="coerce"))
    )
    out.index = pd.DatetimeIndex(dates[~bad].to_numpy(), name="date")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out.astype("float64")


def fetch_mof(cfg: MofConfig, start: dt.date, raw_dir: Path) -> tuple[pd.DataFrame, list[RawFile]]:
    """Fetch history + current year, merge (current year wins on overlap), select tenors."""
    frames: list[pd.DataFrame] = []
    recs: list[RawFile] = []
    for label, url in (("mof:all", cfg.all_history_url), ("mof:current", cfg.current_year_url)):
        dest = raw_dir / "mof" / Path(url).name
        try:
            data, rec = download_raw(label, url, dest)
        except OSError as exc:
            if label == "mof:current":
                log.warning("MoF current-year file unavailable (%s); using history only", exc)
                continue
            raise
        frames.append(parse_mof_csv(decode_bytes(data, cfg.encoding_candidates)))
        recs.append(rec)
    merged = pd.concat(frames).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    missing = [t for t in cfg.tenors.values() if t.upper() not in merged.columns]
    if missing:
        raise ValueError(f"MoF CSV lacks tenor columns {missing}; have {list(merged.columns)}")
    out = pd.DataFrame({name: merged[t.upper()] for name, t in cfg.tenors.items()})
    out = out.loc[out.index >= pd.Timestamp(start)]
    out.index.name = "date"
    log.info(
        "MoF JGB: %d rows, %s -> %s, NaN per col %s",
        len(out),
        out.index.min().date(),
        out.index.max().date(),
        out.isna().sum().to_dict(),
    )
    return out, recs
