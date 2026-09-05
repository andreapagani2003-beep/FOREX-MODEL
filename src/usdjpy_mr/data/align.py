"""Timestamp alignment of FX, UST and JGB series onto one daily snapshot calendar.

Conventions (config `alignment.convention`):

ny_close   Snapshot at 17:00 New York on weekday t. FX close and UST CMT yields are the
           observations of t itself (both are NY-close quantities). The JGB observation of t
           was fixed at ~15:00 Tokyo, i.e. ~11 hours *before* the snapshot, so it is also the
           value of t; on a Japan holiday the most recent earlier Tokyo close is used (as-of)
           and the lag in calendar days is recorded in `jgb_lag_days`.
tokyo_close Snapshot at 15:00 Tokyo on weekday t. JGB is the observation of t. UST and FX are
           the NY closes of t-1 (the latest available before the snapshot); on a US holiday the
           most recent earlier NY close is used and `us_lag_days` / `fx_lag_days` record it.

Rules shared by both conventions:
- Columns in `drop_if_missing` must be observed on their own snapshot date; otherwise the row
  is dropped and the date is reported in the gap log.
- Columns in `asof_max_lag_days` are carried forward from the last observation, but only up to
  the configured lag; beyond that the value is NaN, the row is dropped, and it is logged.
- Nothing is ever forward-filled silently: every carried-forward value has a lag column, and
  every dropped date is returned in `AlignmentReport`.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from usdjpy_mr.config import AlignmentConfig
from usdjpy_mr.data.calendars import business_days, classify_missing
from usdjpy_mr.utils.logging import get_logger

log = get_logger(__name__)

OUTPUT_COLUMNS = ["usdjpy", "us2y", "us10y", "jgb2y", "jgb10y", "spread2y", "spread10y"]
OPTIONAL_COLUMNS = ["us10y_real"]
LAG_COLUMNS = ["fx_lag_days", "us_lag_days", "jgb_lag_days"]

CONVENTION_NOTES = {
    "ny_close": (
        "Snapshot 17:00 America/New_York on date t. usdjpy, us2y, us10y, us10y_real: observation "
        "of t. jgb2y, jgb10y: Tokyo 15:00 close of t (precedes snapshot by ~11h); on Japan "
        "holidays the previous Tokyo close, lag recorded in jgb_lag_days."
    ),
    "tokyo_close": (
        "Snapshot 15:00 Asia/Tokyo on date t. jgb2y, jgb10y: observation of t. usdjpy, us2y, "
        "us10y, us10y_real: NY close of the previous weekday (t-1 or earlier on US holidays), "
        "lag recorded in fx_lag_days / us_lag_days."
    ),
}

_COL_GROUP = {
    "usdjpy": "fx",
    "us2y": "us",
    "us10y": "us",
    "us10y_real": "us",
    "jgb2y": "jgb",
    "jgb10y": "jgb",
}


@dataclass
class AlignmentReport:
    convention: str
    calendar_start: str
    calendar_end: str
    calendar_days: int
    rows_out: int
    dropped: pd.DataFrame = field(default_factory=pd.DataFrame)  # date, column, reason
    carried_forward: dict[str, int] = field(default_factory=dict)  # column -> n rows with lag>0
    max_lag_days: dict[str, int] = field(default_factory=dict)
    source_gaps: dict[str, pd.DataFrame] = field(default_factory=dict)  # per source: date,reason

    def as_dict(self) -> dict:
        return {
            "convention": self.convention,
            "convention_note": CONVENTION_NOTES[self.convention],
            "calendar_start": self.calendar_start,
            "calendar_end": self.calendar_end,
            "calendar_days": self.calendar_days,
            "rows_out": self.rows_out,
            "rows_dropped": int(self.dropped["date"].nunique()) if len(self.dropped) else 0,
            "dropped_by_reason": (
                {
                    f"{col}/{reason}": int(n)
                    for (col, reason), n in self.dropped.groupby(["column", "reason"])
                    .size()
                    .items()
                }
                if len(self.dropped)
                else {}
            ),
            "carried_forward_rows": self.carried_forward,
            "max_lag_days": self.max_lag_days,
            "source_gaps": {
                k: {
                    "holiday": int((v["reason"] == "holiday").sum()),
                    "publication_lag": int((v["reason"] == "publication_lag").sum()),
                    "unexplained": int((v["reason"] == "unexplained").sum()),
                    "unexplained_dates": [
                        d.strftime("%Y-%m-%d") for d in v.loc[v["reason"] == "unexplained", "date"]
                    ],
                }
                for k, v in self.source_gaps.items()
            },
        }


def asof_with_lag(
    s: pd.Series, index: pd.DatetimeIndex, max_lag_days: int, shift_days: int = 0
) -> tuple[pd.Series, pd.Series]:
    """Value of `s` as of (index - shift_days), carried forward at most `max_lag_days` days.

    Returns (values, lag_days). lag_days is measured from the *target* date (index - shift) to
    the observation date used; NaN where no observation within the window.
    """
    s = s.dropna().sort_index()
    target = index - pd.Timedelta(days=shift_days)
    obs = pd.DataFrame({"obs_date": s.index, "value": s.to_numpy()})
    tgt = pd.DataFrame({"target": target})
    merged = pd.merge_asof(
        tgt,
        obs,
        left_on="target",
        right_on="obs_date",
        direction="backward",
        tolerance=pd.Timedelta(days=max_lag_days),
    )
    lag = (merged["target"] - merged["obs_date"]).dt.days
    values = pd.Series(merged["value"].to_numpy(), index=index, dtype="float64")
    lag = pd.Series(lag.to_numpy(), index=index, dtype="float64")
    return values, lag


def _exact(
    s: pd.Series, index: pd.DatetimeIndex, shift_days: int = 0
) -> tuple[pd.Series, pd.Series]:
    """Observation on exactly (index - shift_days); NaN otherwise. Lag is 0 or NaN."""
    target = index - pd.Timedelta(days=shift_days)
    vals = s.dropna().reindex(target).to_numpy()
    values = pd.Series(vals, index=index, dtype="float64")
    lag = pd.Series(0.0, index=index).where(values.notna())
    return values, lag


def _previous_weekday(index: pd.DatetimeIndex, weekmask: str) -> pd.Series:
    """Calendar-day distance from each index date back to the previous weekday (1..3)."""
    mask = "".join(
        "1" if d in weekmask.split() else "0"
        for d in ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
    )
    days = index.to_numpy().astype("datetime64[D]")
    prev = np.busday_offset(days, -1, roll="backward", weekmask=mask)
    return pd.Series((days - prev).astype(int), index=index)


def align_daily(
    sources: dict[str, pd.Series],
    cfg: AlignmentConfig,
    start: dt.date,
    end: dt.date,
) -> tuple[pd.DataFrame, AlignmentReport]:
    """Align raw series (usdjpy, us2y, us10y, [us10y_real], jgb2y, jgb10y) onto one calendar."""
    required = ["usdjpy", "us2y", "us10y", "jgb2y", "jgb10y"]
    missing = [c for c in required if c not in sources]
    if missing:
        raise ValueError(f"align_daily: missing source series {missing}")
    unknown = [c for c in sources if c not in _COL_GROUP]
    if unknown:
        raise ValueError(f"align_daily: unknown source series {unknown}")

    index = business_days(start, end, cfg.weekmask)
    out = pd.DataFrame(index=index)
    lags: dict[str, pd.Series] = {}
    prev_wd = _previous_weekday(index, cfg.weekmask)

    for col, s in sources.items():
        group = _COL_GROUP[col]
        # Which date does this column's observation come from, relative to the snapshot date?
        if cfg.convention == "ny_close":
            base_shift = pd.Series(0, index=index)
        else:  # tokyo_close: JGB is same-day, NY-close quantities are previous weekday
            base_shift = pd.Series(0, index=index) if group == "jgb" else prev_wd

        if col in cfg.drop_if_missing:
            # exact match on the per-row target date
            target = index - pd.to_timedelta(base_shift.to_numpy(), unit="D")
            vals = s.dropna().reindex(target).to_numpy()
            values = pd.Series(vals, index=index, dtype="float64")
            lag = pd.Series(base_shift.to_numpy(), index=index, dtype="float64").where(
                values.notna()
            )
        elif col in cfg.asof_max_lag_days:
            max_lag = cfg.asof_max_lag_days[col]
            if (base_shift == base_shift.iloc[0]).all():
                values, lag = asof_with_lag(s, index, max_lag, int(base_shift.iloc[0]))
            else:
                # per-row shift: compute as-of against shifted targets directly
                target = index - pd.to_timedelta(base_shift.to_numpy(), unit="D")
                v, lag0 = asof_with_lag(s, pd.DatetimeIndex(target), max_lag, 0)
                values = pd.Series(v.to_numpy(), index=index, dtype="float64")
                lag = pd.Series(lag0.to_numpy() + base_shift.to_numpy(), index=index).where(
                    values.notna()
                )
        else:
            raise ValueError(
                f"column {col!r} is neither in drop_if_missing nor asof_max_lag_days; be explicit"
            )
        out[col] = values
        lags[col] = lag

    # Lag columns per group (max over the group's columns).
    for group, lag_col in (("fx", "fx_lag_days"), ("us", "us_lag_days"), ("jgb", "jgb_lag_days")):
        members = [c for c in sources if _COL_GROUP[c] == group and c != "us10y_real"]
        out[lag_col] = pd.concat([lags[c] for c in members], axis=1).max(axis=1)

    # Drop rows with NaN in required columns and record why.
    drop_records: list[tuple[pd.Timestamp, str, str]] = []
    for col in required:
        nan_idx = out.index[out[col].isna()]
        reason = (
            "missing_on_snapshot_date" if col in cfg.drop_if_missing else "stale_beyond_max_lag"
        )
        drop_records.extend((d, col, reason) for d in nan_idx)
    dropped = pd.DataFrame(drop_records, columns=["date", "column", "reason"])
    keep = out[required].notna().all(axis=1)
    out = out.loc[keep].copy()

    out["spread2y"] = out["us2y"] - out["jgb2y"]
    out["spread10y"] = out["us10y"] - out["jgb10y"]
    cols = OUTPUT_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in out.columns] + LAG_COLUMNS
    out = out[cols]
    out.index.name = "date"

    carried = {
        c: int((lags[c].loc[out.index] > (0 if cfg.convention == "ny_close" else 1)).sum())
        for c in sources
        if c in cfg.asof_max_lag_days
    }
    max_lag = {
        c: int(lags[c].loc[out.index].max())
        for c in sources
        if lags[c].loc[out.index].notna().any()
    }

    # Per-source gap classification (weekdays with no observation at all, before alignment).
    source_gaps: dict[str, pd.DataFrame] = {}
    for col, s in sources.items():
        obs_days = s.dropna().index
        missing_days = index.difference(obs_days)
        market = "jp" if _COL_GROUP[col] == "jgb" else "us"
        source_gaps[col] = classify_missing(
            missing_days, market, start, end, cfg.publication_lag_days
        )

    report = AlignmentReport(
        convention=cfg.convention,
        calendar_start=str(index.min().date()),
        calendar_end=str(index.max().date()),
        calendar_days=len(index),
        rows_out=len(out),
        dropped=dropped,
        carried_forward=carried,
        max_lag_days=max_lag,
        source_gaps=source_gaps,
    )
    n_drop = dropped["date"].nunique() if len(dropped) else 0
    log.info(
        "aligned (%s): %d calendar weekdays -> %d rows; %d dates dropped; carried forward %s; "
        "max lag %s",
        cfg.convention,
        len(index),
        len(out),
        n_drop,
        carried,
        max_lag,
    )
    for col, g in source_gaps.items():
        unexplained = g.loc[g["reason"] == "unexplained", "date"]
        if len(unexplained):
            log.warning(
                "%s: %d weekday gaps not explained by holidays, e.g. %s",
                col,
                len(unexplained),
                [d.strftime("%Y-%m-%d") for d in unexplained[:5]],
            )
    return out, report
