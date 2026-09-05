import datetime as dt

import numpy as np
import pandas as pd
import pytest
from tests.conftest import FIX_END, FIX_START, JP_HOLIDAY, US_HOLIDAY, make_config

from usdjpy_mr.data.align import LAG_COLUMNS, OUTPUT_COLUMNS, align_daily, asof_with_lag


def test_ny_close_alignment(cfg, fixture_sources):
    df, rep = align_daily(fixture_sources, cfg.alignment, FIX_START, FIX_END)

    # 15 weekdays; the US holiday has no UST print -> dropped; Japan holiday kept via as-of.
    assert len(df) == 14
    assert US_HOLIDAY not in df.index
    assert JP_HOLIDAY in df.index
    assert list(df.columns) == OUTPUT_COLUMNS + ["us10y_real"] + LAG_COLUMNS

    # Same-day values everywhere except JGB on the Japan holiday (carried from Friday 12th).
    assert df.loc["2024-07-02", "usdjpy"] == pytest.approx(160.02)
    assert df.loc["2024-07-02", "us10y"] == pytest.approx(4.02)
    assert df.loc["2024-07-02", "jgb10y"] == pytest.approx(1.02)
    assert df.loc[JP_HOLIDAY, "jgb10y"] == pytest.approx(1.12)
    assert df.loc[JP_HOLIDAY, "jgb2y"] == pytest.approx(0.12)
    assert df.loc[JP_HOLIDAY, "jgb_lag_days"] == 3
    assert df.loc["2024-07-16", "jgb_lag_days"] == 0
    assert (df["fx_lag_days"] == 0).all()
    assert (df["us_lag_days"] == 0).all()

    # Spreads are exact differences.
    np.testing.assert_allclose(df["spread10y"], df["us10y"] - df["jgb10y"])
    np.testing.assert_allclose(df["spread2y"], df["us2y"] - df["jgb2y"])

    # Report content.
    assert rep.rows_out == 14
    assert rep.calendar_days == 15
    d = rep.dropped
    assert set(d["date"]) == {US_HOLIDAY}
    assert set(d["column"]) == {"us2y", "us10y"}
    assert (d["reason"] == "missing_on_snapshot_date").all()
    assert rep.carried_forward == {"jgb2y": 1, "jgb10y": 1, "us10y_real": 0}
    gaps = rep.as_dict()["source_gaps"]
    assert gaps["us10y"]["holiday"] == 1 and gaps["us10y"]["unexplained"] == 0
    assert gaps["jgb10y"]["holiday"] == 1 and gaps["jgb10y"]["unexplained"] == 0


def test_ny_close_never_uses_future_jgb(cfg, fixture_sources):
    """The JGB value on the Japan holiday must come from before, never from the day after."""
    src = dict(fixture_sources)
    s = src["jgb10y"].copy()
    s.loc[JP_HOLIDAY + pd.Timedelta(days=1)] = 99.0  # poison the next day
    src["jgb10y"] = s.sort_index()
    df, _ = align_daily(src, cfg.alignment, FIX_START, FIX_END)
    assert df.loc[JP_HOLIDAY, "jgb10y"] == pytest.approx(1.12)
    assert df.loc["2024-07-16", "jgb10y"] == 99.0


def test_stale_beyond_max_lag_is_dropped_and_logged(tmp_path, fixture_sources):
    cfg = make_config(
        tmp_path, alignment={"asof_max_lag_days": {"jgb2y": 1, "jgb10y": 1, "us10y_real": 4}}
    )
    df, rep = align_daily(fixture_sources, cfg.alignment, FIX_START, FIX_END)
    assert JP_HOLIDAY not in df.index  # Friday->Monday is a 3-day lag > 1
    assert len(df) == 13
    reasons = rep.dropped.set_index("date")["reason"]
    assert reasons.loc[JP_HOLIDAY].iloc[0] == "stale_beyond_max_lag"


def test_tokyo_close_alignment(cfg_tokyo, fixture_sources):
    df, rep = align_daily(fixture_sources, cfg_tokyo.alignment, FIX_START, FIX_END)

    # JGB mandatory same-day -> Japan holiday dropped; US holiday kept via as-of.
    assert JP_HOLIDAY not in df.index
    assert US_HOLIDAY in df.index
    # July 1 (Mon): previous NY close is Fri Jun 28, which is outside the fixture -> dropped.
    assert pd.Timestamp("2024-07-01") not in df.index
    assert len(df) == 13

    # Tue 2 Jul: FX/UST from Mon 1 Jul (lag 1); JGB from 2 Jul (lag 0).
    assert df.loc["2024-07-02", "usdjpy"] == pytest.approx(160.01)
    assert df.loc["2024-07-02", "us10y"] == pytest.approx(4.01)
    assert df.loc["2024-07-02", "jgb10y"] == pytest.approx(1.02)
    assert df.loc["2024-07-02", "fx_lag_days"] == 1
    assert df.loc["2024-07-02", "us_lag_days"] == 1
    assert df.loc["2024-07-02", "jgb_lag_days"] == 0
    # Thu 4 Jul: UST from Wed 3 Jul (lag 1). Fri 5 Jul: UST from Wed 3 Jul (lag 2), FX from Thu.
    assert df.loc["2024-07-04", "us10y"] == pytest.approx(4.03)
    assert df.loc["2024-07-05", "us10y"] == pytest.approx(4.03)
    assert df.loc["2024-07-05", "us_lag_days"] == 2
    assert df.loc["2024-07-05", "usdjpy"] == pytest.approx(160.04)
    assert df.loc["2024-07-05", "fx_lag_days"] == 1
    # Mon 8 Jul: previous NY close is Fri 5 Jul (lag 3).
    assert df.loc["2024-07-08", "us10y"] == pytest.approx(4.05)
    assert df.loc["2024-07-08", "us_lag_days"] == 3
    assert rep.convention == "tokyo_close"


def test_asof_with_lag_tolerance():
    s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2024-07-01", "2024-07-10"]))
    idx = pd.DatetimeIndex(["2024-07-01", "2024-07-03", "2024-07-09", "2024-07-10"])
    v, lag = asof_with_lag(s, idx, max_lag_days=2)
    assert v.tolist()[:2] == [1.0, 1.0]
    assert np.isnan(v.iloc[2])  # 8 days stale > 2
    assert v.iloc[3] == 2.0
    assert lag.tolist()[0] == 0 and lag.tolist()[1] == 2


def test_missing_source_raises(cfg, fixture_sources):
    src = {k: v for k, v in fixture_sources.items() if k != "jgb2y"}
    with pytest.raises(ValueError, match="missing source"):
        align_daily(src, cfg.alignment, FIX_START, FIX_END)


def test_unknown_source_raises(cfg, fixture_sources):
    src = dict(fixture_sources, bogus=fixture_sources["usdjpy"])
    with pytest.raises(ValueError, match="unknown source"):
        align_daily(src, cfg.alignment, FIX_START, FIX_END)


def test_start_end_window_respected(cfg, fixture_sources):
    df, _ = align_daily(fixture_sources, cfg.alignment, dt.date(2024, 7, 8), dt.date(2024, 7, 12))
    assert df.index.min() == pd.Timestamp("2024-07-08")
    assert df.index.max() == pd.Timestamp("2024-07-12")
    assert len(df) == 5
