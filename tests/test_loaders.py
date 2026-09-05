import datetime as dt

import numpy as np
import pandas as pd
import pytest
from tests.conftest import FIXTURES

from usdjpy_mr.data.fred import parse_fred_csv
from usdjpy_mr.data.fx_yahoo import fetch_fx_yahoo, normalise_yahoo_frame, relabel_by_weekdays
from usdjpy_mr.data.ibkr import bars_to_series
from usdjpy_mr.data.mof import decode_bytes, parse_mof_csv


# --- FRED -------------------------------------------------------------------------------------
def test_parse_fred_csv_dot_is_nan():
    s = parse_fred_csv((FIXTURES / "fred_DGS10_sample.csv").read_text(), "DGS10")
    assert s.index.name == "date"
    assert s.dtype == "float64"
    assert len(s) == 5
    assert np.isnan(s.loc["2024-07-04"])
    assert s.loc["2024-07-05"] == pytest.approx(4.28)


def test_parse_fred_csv_new_header():
    s = parse_fred_csv((FIXTURES / "fred_observation_date_header.csv").read_text(), "DGS2")
    assert s.name == "DGS2"
    assert s.loc["2024-07-01"] == pytest.approx(4.77)


def test_parse_fred_csv_bad_shape():
    with pytest.raises(ValueError):
        parse_fred_csv("DATE\n2024-01-01\n", "DGS2")


# --- MoF --------------------------------------------------------------------------------------
def test_parse_mof_csv_schema():
    text = decode_bytes((FIXTURES / "mof_jgbcme_sample.csv").read_bytes(), ["utf-8", "cp932"])
    df = parse_mof_csv(text)
    assert df.index.name == "date"
    assert list(df.columns[:3]) == ["1Y", "2Y", "3Y"]
    assert "10Y" in df.columns
    assert len(df) == 5
    assert df.loc["2024-07-01", "10Y"] == pytest.approx(1.056)
    assert np.isnan(df.loc["2024-07-03", "1Y"])  # '-' -> NaN
    assert df.loc["2024-07-03", "2Y"] == pytest.approx(0.360)


def test_parse_mof_csv_without_title_line():
    text = "Date,2Y,10Y\n2024/7/1,0.35,1.05\n2024/7/2,0.36,1.06\n"
    df = parse_mof_csv(text)
    assert df.loc["2024-07-02", "10Y"] == pytest.approx(1.06)


def test_parse_mof_csv_no_header():
    with pytest.raises(ValueError):
        parse_mof_csv("nothing,here\n1,2\n")


def test_decode_bytes_cp932():
    data = "Date,2Y\n2024/7/1,0.35\n".encode("cp932") + "円".encode("cp932")
    assert "Date" in decode_bytes(data, ["utf-8", "cp932"])


# --- Yahoo ------------------------------------------------------------------------------------
def _yahoo_multiindex_frame() -> pd.DataFrame:
    idx = pd.to_datetime(["2024-07-01", "2024-07-02", "2024-07-02", "2024-07-03"])
    cols = pd.MultiIndex.from_product([["Open", "Close"], ["USDJPY=X"]], names=["Price", "Ticker"])
    data = np.array([[161.0, 161.5], [161.5, 161.7], [161.5, 161.8], [161.8, np.nan]])
    return pd.DataFrame(data, index=idx, columns=cols)


def test_normalise_yahoo_multiindex_dedup_and_dropna():
    s = normalise_yahoo_frame(_yahoo_multiindex_frame(), "Close", "USDJPY=X")
    assert s.name == "usdjpy"
    assert s.index.name == "date"
    assert list(s.index.strftime("%Y-%m-%d")) == ["2024-07-01", "2024-07-02"]
    assert s.loc["2024-07-02"] == pytest.approx(161.8)  # keep last duplicate


def test_normalise_yahoo_flat_columns_tz_aware():
    idx = pd.to_datetime(["2024-07-01 00:00", "2024-07-02 00:00"]).tz_localize("Europe/London")
    df = pd.DataFrame({"Close": [161.0, 161.5]}, index=idx)
    s = normalise_yahoo_frame(df, "Close", "USDJPY=X")
    assert s.index.tz is None
    assert s.iloc[1] == pytest.approx(161.5)


def test_normalise_yahoo_empty():
    with pytest.raises(ValueError):
        normalise_yahoo_frame(pd.DataFrame(), "Close", "USDJPY=X")


def test_fetch_fx_yahoo_archives_raw(cfg, tmp_path):
    calls = {}

    def fake_download(ticker: str, start: dt.date, end: dt.date | None) -> pd.DataFrame:
        calls["args"] = (ticker, start, end)
        return _yahoo_multiindex_frame()

    s, rec = fetch_fx_yahoo(cfg.sources.fx, dt.date(2024, 7, 1), None, cfg.raw_dir, fake_download)
    assert calls["args"][0] == "USDJPY=X"
    assert len(s) == 2
    assert rec.source == "yahoo:USDJPY=X"
    assert (tmp_path / "data/raw/yahoo/USDJPY_X.csv").exists()
    assert rec.sha256 and rec.bytes > 0


# --- IBKR (pure conversion only; no TWS) -------------------------------------------------------
class _Bar:
    def __init__(self, date, close):
        self.date, self.close = date, close


def test_bars_to_series():
    bars = [_Bar(dt.date(2024, 7, 1), 161.0), _Bar(dt.date(2024, 7, 2), 161.5)]
    s = bars_to_series(bars)
    assert s.name == "usdjpy"
    assert s.loc["2024-07-02"] == pytest.approx(161.5)
    with pytest.raises(ValueError):
        bars_to_series([])


def test_relabel_by_weekdays_moves_monday_to_friday():
    idx = pd.to_datetime(
        ["2026-08-28", "2026-08-31", "2026-09-01", "2026-09-05"]
    )  # Fri Mon Tue Sat
    s = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx, name="usdjpy")
    out = relabel_by_weekdays(s, -1)
    assert list(out.index.strftime("%Y-%m-%d")) == [
        "2026-08-27",
        "2026-08-28",
        "2026-08-31",
        "2026-09-03",
    ]
    assert out.loc["2026-08-28"] == 2.0  # Monday's row is Friday's close
    assert relabel_by_weekdays(s, 0) is s


def test_fetch_fx_yahoo_applies_offset(cfg):
    def fake_download(ticker, start, end):
        idx = pd.to_datetime(["2026-08-31", "2026-09-01"])
        return pd.DataFrame({"Close": [160.12, 159.75]}, index=idx)

    s, _ = fetch_fx_yahoo(cfg.sources.fx, dt.date(2026, 8, 1), None, cfg.raw_dir, fake_download)
    assert cfg.sources.fx.yahoo_label_offset_bdays == -1
    assert s.loc["2026-08-28"] == pytest.approx(160.12)
    assert s.loc["2026-08-31"] == pytest.approx(159.75)
