from __future__ import annotations

import copy
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from usdjpy_mr.config import Config

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        out[k] = (
            _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
        )
    return out


def make_config(root: Path, config_name: str = "default.yaml", **overrides) -> Config:
    raw = yaml.safe_load((REPO_ROOT / "configs" / config_name).read_text(encoding="utf-8"))
    raw = _deep_merge(raw, overrides)
    raw["root"] = root
    return Config.model_validate(raw)


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return make_config(tmp_path)


@pytest.fixture
def cfg_tokyo(tmp_path: Path) -> Config:
    return make_config(tmp_path, "tokyo_close.yaml")


# --- Hand-built alignment fixture: 1-19 July 2024 ---------------------------------------------
# 2024-07-04 Thu: US Independence Day (no UST print; FX trades).
# 2024-07-15 Mon: Japan Marine Day (no JGB print).
# Values encode their own date: value = base + day/100, so provenance is visible in asserts.
FIX_START = dt.date(2024, 7, 1)
FIX_END = dt.date(2024, 7, 19)
US_HOLIDAY = pd.Timestamp("2024-07-04")
JP_HOLIDAY = pd.Timestamp("2024-07-15")


def _coded(base: float, days: pd.DatetimeIndex) -> pd.Series:
    s = pd.Series([base + d.day / 100 for d in days], index=days, dtype="float64")
    s.index.name = "date"
    return s


@pytest.fixture
def fixture_sources() -> dict[str, pd.Series]:
    weekdays = pd.bdate_range(FIX_START, FIX_END)
    us_days = weekdays.drop(US_HOLIDAY)
    jp_days = weekdays.drop(JP_HOLIDAY)
    return {
        "usdjpy": _coded(160.0, weekdays),
        "us2y": _coded(4.0, us_days),
        "us10y": _coded(4.0, us_days),
        "us10y_real": _coded(2.0, us_days),
        "jgb2y": _coded(0.0, jp_days),
        "jgb10y": _coded(1.0, jp_days),
    }


# --- Synthetic long daily frame that should pass validation ------------------------------------
@pytest.fixture
def synthetic_daily() -> pd.DataFrame:
    idx = pd.bdate_range("2010-01-04", "2024-12-31")
    idx.name = "date"
    rng = np.random.default_rng(0)
    n = len(idx)
    df = pd.DataFrame(index=idx)
    # spot co-moves with the US 10y on the same day (as real data does), so the timing check passes
    shock = rng.normal(0, 1, n)
    df["us2y"] = 1.5 + np.cumsum(rng.normal(0, 0.02, n)).clip(-1.4, 4)
    df["us10y"] = 2.5 + np.cumsum(0.02 * shock).clip(-1.9, 3)
    df["usdjpy"] = 100 + np.cumsum(0.4 * shock + rng.normal(0, 0.3, n)).clip(-25, 60)
    df["jgb2y"] = 0.1 + np.cumsum(rng.normal(0, 0.005, n)).clip(-0.4, 1)
    df["jgb10y"] = 0.5 + np.cumsum(rng.normal(0, 0.005, n)).clip(-0.7, 1.5)
    df["spread2y"] = df["us2y"] - df["jgb2y"]
    df["spread10y"] = df["us10y"] - df["jgb10y"]
    df["us10y_real"] = df["us10y"] - 2.0
    df["fx_lag_days"] = 0.0
    df["us_lag_days"] = 0.0
    df["jgb_lag_days"] = 0.0
    return df
