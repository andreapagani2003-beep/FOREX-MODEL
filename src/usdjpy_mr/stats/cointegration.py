"""Cointegration: Engle-Granger (both orderings), Johansen, rolling Engle-Granger."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from usdjpy_mr.config import EngleGrangerConfig, JohansenConfig, RollingCointConfig


@dataclass(frozen=True)
class EngleGrangerResult:
    dependent: str
    regressor: str
    statistic: float
    pvalue: float
    crit: dict[str, float]
    alpha: float
    beta: float
    nobs: int

    def as_dict(self) -> dict:
        return {
            "dependent": self.dependent,
            "regressor": self.regressor,
            "statistic": self.statistic,
            "pvalue": self.pvalue,
            "alpha": self.alpha,
            "beta": self.beta,
            "nobs": self.nobs,
            **{f"crit_{k}": v for k, v in self.crit.items()},
        }


def engle_granger(y: pd.Series, x: pd.Series, cfg: EngleGrangerConfig) -> EngleGrangerResult:
    """EG test of y on x: OLS y = a + b x + e, ADF on e with MacKinnon cointegration p-values."""
    df = pd.concat([y, x], axis=1).dropna()
    yv, xv = df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()
    stat, p, crit = coint(yv, xv, trend=cfg.trend, autolag=cfg.autolag, maxlag=cfg.max_lag)
    ols = sm.OLS(yv, sm.add_constant(xv)).fit()
    return EngleGrangerResult(
        dependent=str(y.name),
        regressor=str(x.name),
        statistic=float(stat),
        pvalue=float(p),
        crit={"1": float(crit[0]), "5": float(crit[1]), "10": float(crit[2])},
        alpha=float(ols.params[0]),
        beta=float(ols.params[1]),
        nobs=len(df),
    )


def engle_granger_both(
    y: pd.Series, x: pd.Series, cfg: EngleGrangerConfig
) -> dict[str, EngleGrangerResult]:
    return {"y_on_x": engle_granger(y, x, cfg), "x_on_y": engle_granger(x, y, cfg)}


@dataclass(frozen=True)
class JohansenResult:
    trace_stat: list[float]
    trace_crit: list[list[float]]  # per rank: [90, 95, 99]
    max_eig_stat: list[float]
    max_eig_crit: list[list[float]]
    eigenvalues: list[float]
    cointegrating_vector: list[float]  # normalised so the first variable has coefficient 1
    rank_trace_5pct: int
    rank_maxeig_5pct: int
    nobs: int

    def as_dict(self) -> dict:
        return {
            "trace_r0": self.trace_stat[0],
            "trace_r0_crit95": self.trace_crit[0][1],
            "trace_r1": self.trace_stat[1],
            "trace_r1_crit95": self.trace_crit[1][1],
            "maxeig_r0": self.max_eig_stat[0],
            "maxeig_r0_crit95": self.max_eig_crit[0][1],
            "maxeig_r1": self.max_eig_stat[1],
            "maxeig_r1_crit95": self.max_eig_crit[1][1],
            "rank_trace_5pct": self.rank_trace_5pct,
            "rank_maxeig_5pct": self.rank_maxeig_5pct,
            "beta_implied": -self.cointegrating_vector[1],
            "nobs": self.nobs,
        }


def johansen(y: pd.Series, x: pd.Series, cfg: JohansenConfig) -> JohansenResult:
    """Johansen trace / max-eigenvalue tests on [y, x]. Vector normalised on y so that the implied
    long-run relation is y = beta_implied * x (+ deterministic terms)."""
    df = pd.concat([y, x], axis=1).dropna()
    with warnings.catch_warnings():
        # statsmodels casts near-real complex eigenvalues internally; we take real parts below
        warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
        res = coint_johansen(df.to_numpy(), det_order=cfg.det_order, k_ar_diff=cfg.k_ar_diff)
    vec = np.real(res.evec[:, 0])
    vec = vec / vec[0]
    eig = np.real(res.eig)

    def _rank(stats: np.ndarray, crit: np.ndarray) -> int:
        r = 0
        for i in range(len(stats)):
            if stats[i] > crit[i, 1]:  # 95% column
                r = i + 1
            else:
                break
        return r

    return JohansenResult(
        trace_stat=[float(v) for v in res.lr1],
        trace_crit=[[float(c) for c in row] for row in res.cvt],
        max_eig_stat=[float(v) for v in res.lr2],
        max_eig_crit=[[float(c) for c in row] for row in res.cvm],
        eigenvalues=[float(v) for v in eig],
        cointegrating_vector=[float(v) for v in vec],
        rank_trace_5pct=_rank(res.lr1, res.cvt),
        rank_maxeig_5pct=_rank(res.lr2, res.cvm),
        nobs=len(df),
    )


def rolling_engle_granger(
    y: pd.Series, x: pd.Series, eg_cfg: EngleGrangerConfig, cfg: RollingCointConfig
) -> pd.DataFrame:
    """EG statistic and p-value on a rolling window ending at each date (uses data up to t only)."""
    df = pd.concat([y, x], axis=1).dropna()
    yv, xv = df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()
    rows = []
    for end in range(cfg.window, len(df) + 1, cfg.step):
        sl = slice(end - cfg.window, end)
        stat, p, crit = coint(
            yv[sl], xv[sl], trend=eg_cfg.trend, autolag=eg_cfg.autolag, maxlag=eg_cfg.max_lag
        )
        rows.append((df.index[end - 1], float(stat), float(p), float(crit[1])))
    out = pd.DataFrame(rows, columns=["date", "statistic", "pvalue", "crit_5"]).set_index("date")
    out["reject_5pct"] = out["pvalue"] < 0.05
    return out
