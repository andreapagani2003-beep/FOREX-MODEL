import datetime as dt

import numpy as np
import pandas as pd

from usdjpy_mr.data.validate import validate_daily

TODAY = dt.date(2025, 1, 3)


def _cfg(cfg):
    # synthetic frame ends 2024-12-31; pin the end date so staleness is judged against it
    return cfg.model_copy(update={"project": cfg.project.model_copy(update={"end_date": TODAY})})


def test_synthetic_passes(cfg, synthetic_daily):
    res = validate_daily(synthetic_daily, _cfg(cfg), today=TODAY)
    assert res.ok, [f.message for f in res.errors]
    assert res.stats["rows"] == len(synthetic_daily)
    assert res.stats["largest_gap_weekdays"] == 0
    assert "PASS" in res.to_markdown()


def _checks(res):
    return {f.check for f in res.errors}


def test_catches_nan(cfg, synthetic_daily):
    df = synthetic_daily.copy()
    df.iloc[100, df.columns.get_loc("us10y")] = np.nan
    assert "nan" in _checks(validate_daily(df, _cfg(cfg), today=TODAY))


def test_catches_duplicate_dates(cfg, synthetic_daily):
    df = pd.concat([synthetic_daily, synthetic_daily.iloc[[500]]]).sort_index()
    assert "dates" in _checks(validate_daily(df, _cfg(cfg), today=TODAY))


def test_catches_out_of_bounds(cfg, synthetic_daily):
    df = synthetic_daily.copy()
    df.iloc[10, df.columns.get_loc("usdjpy")] = 250.0
    df.iloc[11, df.columns.get_loc("jgb10y")] = -5.0
    res = validate_daily(df, _cfg(cfg), today=TODAY)
    msgs = [f.message for f in res.errors if f.check == "bounds"]
    assert len(msgs) == 2 and any("usdjpy" in m for m in msgs) and any("jgb10y" in m for m in msgs)


def test_catches_large_gap(cfg, synthetic_daily):
    df = synthetic_daily.drop(synthetic_daily.index[1000:1010])
    res = validate_daily(df, _cfg(cfg), today=TODAY)
    assert "gaps" in _checks(res)
    assert res.stats["largest_gap_weekdays"] == 10


def test_small_gap_is_tolerated(cfg, synthetic_daily):
    df = synthetic_daily.drop(synthetic_daily.index[1000:1003])
    res = validate_daily(df, _cfg(cfg), today=TODAY)
    assert "gaps" not in _checks(res)


def test_catches_spread_inconsistency(cfg, synthetic_daily):
    df = synthetic_daily.copy()
    df.iloc[5, df.columns.get_loc("spread10y")] += 0.5
    assert "consistency" in _checks(validate_daily(df, _cfg(cfg), today=TODAY))


def test_catches_missing_column(cfg, synthetic_daily):
    df = synthetic_daily.drop(columns=["jgb2y"])
    assert "schema" in _checks(validate_daily(df, _cfg(cfg), today=TODAY))


def test_catches_short_history_and_stale_end(cfg, synthetic_daily):
    df = synthetic_daily.loc["2015-01-01":"2020-12-31"]
    res = validate_daily(df, _cfg(cfg), today=TODAY)
    checks = _checks(res)
    assert "coverage" in checks


def test_metadata_row_count_mismatch(cfg, synthetic_daily):
    res = validate_daily(synthetic_daily, _cfg(cfg), metadata={"row_count": 1}, today=TODAY)
    assert "metadata" in _checks(res)
