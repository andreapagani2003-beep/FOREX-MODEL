import datetime as dt

import pandas as pd

from usdjpy_mr.data.calendars import business_days, classify_missing, jp_holidays, us_holidays


def test_us_holidays_known_dates():
    hol = us_holidays(dt.date(2024, 1, 1), dt.date(2024, 12, 31))
    assert dt.date(2024, 7, 4) in hol
    assert dt.date(2024, 11, 28) in hol  # Thanksgiving
    assert dt.date(2024, 7, 5) not in hol


def test_jp_holidays_known_dates_and_market_closures():
    hol = jp_holidays(dt.date(2024, 1, 1), dt.date(2024, 12, 31))
    assert dt.date(2024, 7, 15) in hol  # Marine Day
    assert dt.date(2024, 5, 3) in hol  # Constitution Day
    assert dt.date(2024, 1, 2) in hol  # market closure, not a public holiday
    assert dt.date(2024, 12, 31) in hol
    assert dt.date(2024, 7, 4) not in hol


def test_business_days_weekdays_only():
    idx = business_days(dt.date(2024, 7, 1), dt.date(2024, 7, 7))
    assert len(idx) == 5
    assert idx.name == "date"


def test_classify_missing():
    missing = pd.DatetimeIndex(["2024-07-04", "2024-07-10"])
    out = classify_missing(missing, "us", dt.date(2024, 7, 1), dt.date(2024, 7, 31))
    assert out.set_index("date").loc["2024-07-04", "reason"] == "holiday"
    assert out.set_index("date").loc["2024-07-10", "reason"] == "unexplained"
