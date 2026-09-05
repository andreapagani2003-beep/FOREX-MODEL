"""Unit-root tests. ADF (H0: unit root) and KPSS (H0: stationary) are read together:
ADF reject + KPSS not reject -> I(0); ADF not reject + KPSS reject -> I(1); otherwise ambiguous."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

from usdjpy_mr.config import AdfConfig, KpssConfig


@dataclass(frozen=True)
class UnitRootResult:
    test: str
    statistic: float
    pvalue: float
    lags: int
    nobs: int
    crit: dict[str, float]

    def as_dict(self) -> dict:
        return {
            "test": self.test,
            "statistic": self.statistic,
            "pvalue": self.pvalue,
            "lags": self.lags,
            "nobs": self.nobs,
            **{f"crit_{k}": v for k, v in self.crit.items()},
        }


def adf_test(x: pd.Series | np.ndarray, cfg: AdfConfig) -> UnitRootResult:
    arr = np.asarray(pd.Series(x).dropna(), dtype=float)
    stat, p, lags, nobs, crit, _ = adfuller(
        arr, regression=cfg.regression, autolag=cfg.autolag, result_object=False
    )
    return UnitRootResult(
        "adf",
        float(stat),
        float(p),
        int(lags),
        int(nobs),
        {k.rstrip("%"): float(v) for k, v in crit.items()},
    )


def kpss_test(x: pd.Series | np.ndarray, cfg: KpssConfig) -> UnitRootResult:
    arr = np.asarray(pd.Series(x).dropna(), dtype=float)
    with warnings.catch_warnings():
        # statsmodels warns when the p-value is outside the tabulated range; the value is then
        # capped at 0.01 / 0.10, which is exactly what we want to report.
        warnings.simplefilter("ignore", InterpolationWarning)
        stat, p, lags, crit = kpss(
            arr, regression=cfg.regression, nlags=cfg.nlags, result_object=False
        )
    return UnitRootResult(
        "kpss",
        float(stat),
        float(p),
        int(lags),
        len(arr),
        {k.rstrip("%"): float(v) for k, v in crit.items()},
    )


def integration_order(x: pd.Series, adf_cfg: AdfConfig, kpss_cfg: KpssConfig, alpha: float) -> dict:
    """Classify a series as I(0), I(1) or 'ambiguous' from ADF+KPSS on levels and differences."""
    lvl_adf, lvl_kpss = adf_test(x, adf_cfg), kpss_test(x, kpss_cfg)
    dx = pd.Series(x).diff().dropna()
    dif_adf, dif_kpss = adf_test(dx, adf_cfg), kpss_test(dx, kpss_cfg)

    def _verdict(a: UnitRootResult, k: UnitRootResult) -> str:
        adf_rej, kpss_rej = a.pvalue < alpha, k.pvalue < alpha
        if adf_rej and not kpss_rej:
            return "stationary"
        if not adf_rej and kpss_rej:
            return "unit_root"
        return "ambiguous"

    levels, diffs = _verdict(lvl_adf, lvl_kpss), _verdict(dif_adf, dif_kpss)
    if levels == "stationary":
        order = "I(0)"
    elif levels == "unit_root" and diffs == "stationary":
        order = "I(1)"
    elif levels == "ambiguous" and diffs == "stationary":
        order = "I(1)?"  # levels ambiguous but differences clean: treat as I(1) with a flag
    else:
        order = "ambiguous"
    return {
        "order": order,
        "levels": {"adf": lvl_adf.as_dict(), "kpss": lvl_kpss.as_dict(), "verdict": levels},
        "diffs": {"adf": dif_adf.as_dict(), "kpss": dif_kpss.as_dict(), "verdict": diffs},
    }
