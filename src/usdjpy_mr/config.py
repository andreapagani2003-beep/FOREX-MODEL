"""Typed configuration. Every tunable lives in configs/*.yaml and is validated here."""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
ROOT_ENV = "USDJPY_MR_ROOT"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProjectConfig(_Strict):
    name: str
    start_date: dt.date
    end_date: dt.date | None = None

    def effective_end_date(self) -> dt.date:
        return self.end_date or dt.date.today()


class PathsConfig(_Strict):
    raw_dir: Path
    processed_dir: Path
    reports_dir: Path
    signals_dir: Path


class FredConfig(_Strict):
    api_key_env: str
    csv_url_template: str
    series: dict[str, str]
    snapshot_note: str

    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) or None


class MofConfig(_Strict):
    all_history_url: str
    current_year_url: str
    encoding_candidates: list[str]
    tenors: dict[str, str]
    snapshot_note: str


class FxConfig(_Strict):
    provider: Literal["yahoo", "ibkr"]
    yahoo_ticker: str
    yahoo_field: str
    snapshot_note: str


class IbkrContract(_Strict):
    symbol: str
    currency: str | None = None
    exchange: str
    what_to_show: str = "MIDPOINT"


class IbkrConfig(_Strict):
    host_env: str
    port_env: str
    client_id_env: str
    spot: IbkrContract
    futures: IbkrContract
    bar_size: str
    use_rth: bool
    live_account_confirmed: bool = False

    def host(self) -> str:
        return os.environ.get(self.host_env, "127.0.0.1")

    def port(self) -> int:
        return int(os.environ.get(self.port_env, "7497"))

    def client_id(self) -> int:
        return int(os.environ.get(self.client_id_env, "11"))


class SourcesConfig(_Strict):
    fred: FredConfig
    mof: MofConfig
    fx: FxConfig
    ibkr: IbkrConfig


class AlignmentConfig(_Strict):
    convention: Literal["ny_close", "tokyo_close"]
    drop_if_missing: list[str]
    asof_max_lag_days: dict[str, int]
    weekmask: str = "Mon Tue Wed Thu Fri"

    @field_validator("asof_max_lag_days")
    @classmethod
    def _positive(cls, v: dict[str, int]) -> dict[str, int]:
        bad = {k: n for k, n in v.items() if n < 0}
        if bad:
            raise ValueError(f"asof_max_lag_days must be >= 0: {bad}")
        return v

    @model_validator(mode="after")
    def _disjoint(self) -> AlignmentConfig:
        overlap = set(self.drop_if_missing) & set(self.asof_max_lag_days)
        if overlap:
            raise ValueError(f"columns cannot be both drop_if_missing and as-of: {sorted(overlap)}")
        return self


class ValidationConfig(_Strict):
    bounds: dict[str, tuple[float, float]]
    max_gap_business_days: int = Field(ge=1)
    min_rows: int = Field(ge=1)
    required_start_before: dt.date
    max_days_stale_end: int = Field(ge=0)

    @field_validator("bounds")
    @classmethod
    def _ordered(cls, v: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
        bad = {k: b for k, b in v.items() if b[0] >= b[1]}
        if bad:
            raise ValueError(f"bounds must be [min, max] with min < max: {bad}")
        return v


class Config(_Strict):
    project: ProjectConfig
    paths: PathsConfig
    sources: SourcesConfig
    alignment: AlignmentConfig
    validation: ValidationConfig
    root: Path = Field(default_factory=Path.cwd, exclude=True)

    # Resolved paths -------------------------------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        return self.root / self.paths.raw_dir

    @property
    def processed_dir(self) -> Path:
        return self.root / self.paths.processed_dir

    @property
    def reports_dir(self) -> Path:
        return self.root / self.paths.reports_dir

    @property
    def signals_dir(self) -> Path:
        return self.root / self.paths.signals_dir

    @property
    def daily_parquet(self) -> Path:
        return self.processed_dir / "daily.parquet"

    @property
    def metadata_json(self) -> Path:
        return self.processed_dir / "metadata.json"


def resolve_root(root: str | Path | None = None) -> Path:
    """Project root: explicit arg > $USDJPY_MR_ROOT > cwd. Colab sets the env var after cloning."""
    if root is not None:
        return Path(root).expanduser().resolve()
    env = os.environ.get(ROOT_ENV)
    return Path(env).expanduser().resolve() if env else Path.cwd().resolve()


def load_config(path: str | Path | None = None, root: str | Path | None = None) -> Config:
    """Load and validate a YAML config. Relative `path` is resolved against `root`."""
    root_path = resolve_root(root)
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.is_absolute():
        cfg_path = root_path / cfg_path
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    raw["root"] = root_path
    return Config.model_validate(raw)
