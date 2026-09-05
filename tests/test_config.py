import pytest
from pydantic import ValidationError
from tests.conftest import REPO_ROOT, make_config

from usdjpy_mr.config import load_config


def test_default_config_loads(tmp_path):
    cfg = make_config(tmp_path)
    assert cfg.alignment.convention == "ny_close"
    assert cfg.sources.fred.series["us10y"] == "DGS10"
    assert cfg.daily_parquet == tmp_path / "data/processed/daily.parquet"


def test_tokyo_config_loads(tmp_path):
    cfg = make_config(tmp_path, "tokyo_close.yaml")
    assert cfg.alignment.convention == "tokyo_close"
    assert "usdjpy" in cfg.alignment.asof_max_lag_days


def test_load_config_from_repo():
    cfg = load_config(root=REPO_ROOT)
    assert cfg.root == REPO_ROOT
    assert cfg.sources.ibkr.live_account_confirmed is False


def test_unknown_key_rejected(tmp_path):
    with pytest.raises(ValidationError):
        make_config(tmp_path, alignment={"bogus": 1})


def test_overlapping_alignment_columns_rejected(tmp_path):
    with pytest.raises(ValidationError):
        make_config(tmp_path, alignment={"asof_max_lag_days": {"usdjpy": 2}})


def test_bounds_must_be_ordered(tmp_path):
    with pytest.raises(ValidationError):
        make_config(tmp_path, validation={"bounds": {"usdjpy": [200.0, 70.0]}})


def test_env_root(monkeypatch, tmp_path):
    monkeypatch.setenv("USDJPY_MR_ROOT", str(tmp_path))
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "default.yaml").write_text(
        (REPO_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    cfg = load_config()
    assert cfg.root == tmp_path.resolve()
