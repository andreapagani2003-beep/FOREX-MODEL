import datetime as dt
import json

import pandas as pd
from tests.conftest import FIX_END, FIX_START, make_config

from usdjpy_mr.data.pipeline import build_daily, fetch_all, load_daily
from usdjpy_mr.utils.io import RawFile


def _rec(name: str) -> RawFile:
    return RawFile(
        source=name,
        url=f"test://{name}",
        path=f"/tmp/{name}",
        downloaded_at="now",
        sha256="0" * 64,
        bytes=1,
    )


def test_build_and_load_roundtrip(tmp_path, fixture_sources):
    cfg = make_config(tmp_path, project={"start_date": FIX_START, "end_date": FIX_END})
    df, meta = build_daily(cfg, fixture_sources, [_rec("a"), _rec("b")])
    assert cfg.daily_parquet.exists() and cfg.metadata_json.exists()

    df2, meta2 = load_daily(cfg)
    pd.testing.assert_frame_equal(df, df2)
    assert meta2["row_count"] == 14
    assert meta2["alignment"]["convention"] == "ny_close"
    assert "convention_note" in meta2["alignment"]
    assert meta2["sources"][0]["source"] == "a"
    assert meta2["config"]["alignment"]["convention"] == "ny_close"
    assert meta2["dropped_dates"][0]["date"] == "2024-07-04"
    assert set(meta2["source_rows"]) == set(fixture_sources)
    json.dumps(meta2)  # serialisable


def test_fetch_all_wires_sources(monkeypatch, tmp_path, fixture_sources):
    """fetch_all should call each loader once and return the expected column keys."""
    import usdjpy_mr.data.pipeline as pl

    calls = []

    def fake_fred(fcfg, start, raw_dir):
        calls.append(("fred", start))
        return pd.DataFrame({k: fixture_sources[k] for k in ("us2y", "us10y", "us10y_real")}), [
            _rec("fred")
        ]

    def fake_mof(mcfg, start, raw_dir):
        calls.append(("mof", start))
        return pd.DataFrame({k: fixture_sources[k] for k in ("jgb2y", "jgb10y")}), [_rec("mof")]

    def fake_fx(fxcfg, start, end, raw_dir):
        calls.append(("fx", start))
        return fixture_sources["usdjpy"], _rec("fx")

    monkeypatch.setattr(pl, "fetch_fred_all", fake_fred)
    monkeypatch.setattr(pl, "fetch_mof", fake_mof)
    monkeypatch.setattr(pl, "fetch_fx_yahoo", fake_fx)

    cfg = make_config(tmp_path, project={"start_date": FIX_START, "end_date": FIX_END})
    sources, recs = fetch_all(cfg)
    assert set(sources) == {"usdjpy", "us2y", "us10y", "us10y_real", "jgb2y", "jgb10y"}
    assert [c[0] for c in calls] == ["fred", "mof", "fx"]
    assert all(c[1] == FIX_START - dt.timedelta(days=31) for c in calls)  # warm-up window
    assert len(recs) == 3
