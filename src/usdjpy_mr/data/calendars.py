"""US and Japan holiday calendars, used to explain gaps rather than to fill them.

`holidays` (pure Python) provides official public holidays. Bond-market closures that are not
public holidays (e.g. SIFMA early closes, Japan's Dec 31 / Jan 2-3 market closures) are added
explicitly so that a missing observation is either "known closure" or "unexplained gap".
"""

from __future__ import annotations

import datetime as dt
from functools import lru_cache

import holidays
import pandas as pd

WEEKMASK_DEFAULT = "Mon Tue Wed Thu Fri"


def _year_range(start: dt.date, end: dt.date) -> range:
    return range(start.year, end.year + 1)


@lru_cache(maxsize=8)
def _us_holidays(years: tuple[int, ...]) -> set[dt.date]:
    """US bond-market (SIFMA) full closures: NYSE holidays plus the federal holidays the stock
    market stays open for (Columbus Day, Veterans Day). Union of the two covers the SIFMA list."""
    nyse = holidays.financial_holidays("NYSE", years=years)
    federal = holidays.country_holidays("US", years=years)
    return set(nyse.keys()) | set(federal.keys())


@lru_cache(maxsize=8)
def _jp_holidays(years: tuple[int, ...]) -> set[dt.date]:
    hol = holidays.country_holidays("JP", years=years)
    out = set(hol.keys())
    for y in years:  # JGB market closures around New Year that are not public holidays
        out.update({dt.date(y, 12, 31), dt.date(y, 1, 2), dt.date(y, 1, 3)})
    return out


def us_holidays(start: dt.date, end: dt.date) -> set[dt.date]:
    return {d for d in _us_holidays(tuple(_year_range(start, end))) if start <= d <= end}


def jp_holidays(start: dt.date, end: dt.date) -> set[dt.date]:
    return {d for d in _jp_holidays(tuple(_year_range(start, end))) if start <= d <= end}


def business_days(
    start: dt.date, end: dt.date, weekmask: str = WEEKMASK_DEFAULT
) -> pd.DatetimeIndex:
    """Weekdays between start and end inclusive (no holiday exclusion)."""
    idx = pd.bdate_range(start, end, freq="C", weekmask=weekmask)
    idx.name = "date"
    return idx


def classify_missing(
    missing: pd.DatetimeIndex,
    market: str,
    start: dt.date,
    end: dt.date,
    publication_lag_days: int = 0,
) -> pd.DataFrame:
    """Label each missing weekday as 'holiday' (known closure), 'publication_lag' (within the
    last `publication_lag_days` calendar days before `end`, i.e. not yet published), or
    'unexplained'."""
    hol = us_holidays(start, end) if market == "us" else jp_holidays(start, end)
    lag_from = end - dt.timedelta(days=publication_lag_days)

    def _reason(d: pd.Timestamp) -> str:
        if d.date() in hol:
            return "holiday"
        if publication_lag_days and d.date() > lag_from:
            return "publication_lag"
        return "unexplained"

    rows = [(d, _reason(d)) for d in missing]
    return pd.DataFrame(rows, columns=["date", "reason"])
