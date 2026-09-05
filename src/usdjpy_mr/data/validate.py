"""Validation of processed/daily.parquet against config bounds and structural rules.

Every check appends a `Finding` (level error|warning). Errors fail the phase; warnings are
reported. Nothing is repaired here: the pipeline is re-run instead.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from usdjpy_mr.config import Config
from usdjpy_mr.data.align import LAG_COLUMNS, OPTIONAL_COLUMNS, OUTPUT_COLUMNS
from usdjpy_mr.data.calendars import business_days
from usdjpy_mr.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class Finding:
    level: str  # "error" | "warning"
    check: str
    message: str


@dataclass
class ValidationResult:
    findings: list[Finding] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def error(self, check: str, message: str) -> None:
        self.findings.append(Finding("error", check, message))

    def warn(self, check: str, message: str) -> None:
        self.findings.append(Finding("warning", check, message))

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_markdown(self) -> str:
        lines = [
            "# Phase 1 validation",
            "",
            f"Result: **{'PASS' if self.ok else 'FAIL'}**",
            f"errors: {len(self.errors)}, warnings: {len(self.warnings)}",
            "",
        ]
        if self.stats:
            lines += ["## Stats", "", "| key | value |", "|---|---|"]
            lines += [f"| {k} | {v} |" for k, v in self.stats.items()]
            lines.append("")
        for level in ("error", "warning"):
            items = [f for f in self.findings if f.level == level]
            if items:
                lines += [f"## {level}s", ""]
                lines += [f"- `{f.check}`: {f.message}" for f in items]
                lines.append("")
        return "\n".join(lines)


def validate_daily(
    df: pd.DataFrame,
    cfg: Config,
    metadata: dict | None = None,
    today: dt.date | None = None,
) -> ValidationResult:
    res = ValidationResult()
    today = today or dt.date.today()
    vcfg = cfg.validation

    # -- schema -----------------------------------------------------------------------------
    if df.index.name != "date" or not isinstance(df.index, pd.DatetimeIndex):
        res.error("schema", f"index must be a DatetimeIndex named 'date', got {df.index.name!r}")
    missing = [c for c in OUTPUT_COLUMNS if c not in df.columns]
    if missing:
        res.error("schema", f"missing required columns {missing}")
        return res  # nothing else is meaningful
    extra = [c for c in df.columns if c not in OUTPUT_COLUMNS + OPTIONAL_COLUMNS + LAG_COLUMNS]
    if extra:
        res.warn("schema", f"unexpected extra columns {extra}")

    # -- dates ------------------------------------------------------------------------------
    if df.index.has_duplicates:
        dups = df.index[df.index.duplicated()].unique()
        res.error(
            "dates", f"{len(dups)} duplicate dates, e.g. {dups[:3].strftime('%Y-%m-%d').tolist()}"
        )
    if not df.index.is_monotonic_increasing:
        res.error("dates", "dates are not sorted ascending")
    if (df.index != df.index.normalize()).any():
        res.error("dates", "dates carry a time-of-day component")
    if len(df) < vcfg.min_rows:
        res.error("rows", f"{len(df)} rows < min_rows {vcfg.min_rows}")
    first, last = df.index.min().date(), df.index.max().date()
    if first > vcfg.required_start_before:
        res.error(
            "coverage",
            f"first row {first} is after required_start_before {vcfg.required_start_before}",
        )
    end = cfg.project.effective_end_date()
    if (end - last).days > vcfg.max_days_stale_end:
        res.error("coverage", f"last row {last} is {(end - last).days} days before end_date {end}")

    # -- NaNs -------------------------------------------------------------------------------
    for col in OUTPUT_COLUMNS:
        n = int(df[col].isna().sum())
        if n:
            res.error("nan", f"{col}: {n} NaN values after alignment")
    for col in OPTIONAL_COLUMNS:
        if col in df.columns:
            n = int(df[col].isna().sum())
            if n:
                res.warn("nan", f"{col} (optional): {n} NaN values")

    # -- bounds -----------------------------------------------------------------------------
    for col, (lo, hi) in vcfg.bounds.items():
        if col not in df.columns:
            continue
        s = df[col].dropna()
        bad = s[(s < lo) | (s > hi)]
        if len(bad):
            res.error(
                "bounds",
                f"{col}: {len(bad)} values outside [{lo}, {hi}], "
                f"e.g. {bad.index[0].date()}={bad.iloc[0]:.4f}",
            )

    # -- internal consistency ---------------------------------------------------------------
    for tenor in ("2y", "10y"):
        recomputed = df[f"us{tenor}"] - df[f"jgb{tenor}"]
        if not np.allclose(recomputed, df[f"spread{tenor}"], atol=1e-9, equal_nan=True):
            res.error("consistency", f"spread{tenor} != us{tenor} - jgb{tenor}")
    for col in LAG_COLUMNS:
        if col in df.columns and (df[col] < 0).any():
            res.error("lags", f"{col} has negative values")

    # -- gaps -------------------------------------------------------------------------------
    bdays = business_days(first, last, cfg.alignment.weekmask)
    missing_days = bdays.difference(df.index)
    pos = pd.Series(np.arange(len(bdays)), index=bdays)
    row_pos = pos.reindex(df.index).to_numpy()
    gap_sizes = np.diff(row_pos) - 1  # weekdays skipped between consecutive rows
    big = np.where(gap_sizes > vcfg.max_gap_business_days)[0]
    for i in big[:20]:
        res.error(
            "gaps",
            f"{int(gap_sizes[i])} weekdays missing between "
            f"{df.index[i].date()} and {df.index[i + 1].date()}",
        )
    if len(big) > 20:
        res.error(
            "gaps", f"... and {len(big) - 20} more gaps > {vcfg.max_gap_business_days} weekdays"
        )

    # -- staleness summary (informational) ---------------------------------------------------
    stale: dict[str, float] = {}
    for col in LAG_COLUMNS:
        if col in df.columns:
            stale[col + "_max"] = float(df[col].max())
            stale[col + "_mean"] = round(float(df[col].mean()), 3)

    # -- metadata cross-check ---------------------------------------------------------------
    if metadata is not None:
        if metadata.get("row_count") != len(df):
            res.error("metadata", f"metadata row_count {metadata.get('row_count')} != {len(df)}")
        if metadata.get("alignment", {}).get("convention") != cfg.alignment.convention:
            res.warn("metadata", "metadata alignment convention differs from current config")

    res.stats = {
        "rows": len(df),
        "first_date": str(first),
        "last_date": str(last),
        "weekdays_in_range": len(bdays),
        "weekdays_missing": len(missing_days),
        "coverage_pct": round(100 * len(df) / max(1, len(bdays)), 2),
        "largest_gap_weekdays": int(gap_sizes.max()) if len(gap_sizes) else 0,
        **stale,
        **{f"{c}_min": round(float(df[c].min()), 4) for c in OUTPUT_COLUMNS},
        **{f"{c}_max": round(float(df[c].max()), 4) for c in OUTPUT_COLUMNS},
    }
    for f in res.findings:
        (log.error if f.level == "error" else log.warning)("%s: %s", f.check, f.message)
    log.info(
        "validation %s: %d errors, %d warnings",
        "PASS" if res.ok else "FAIL",
        len(res.errors),
        len(res.warnings),
    )
    return res
