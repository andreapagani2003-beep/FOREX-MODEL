"""Ornstein-Uhlenbeck fit of a residual via its discrete AR(1) form, with a bootstrap CI on the
half-life.

Discrete OU: e_t - e_{t-1} = theta * (mu - e_{t-1}) * dt + sigma * sqrt(dt) * z_t. Regressing
d e_t on e_{t-1} gives slope b = phi - 1 with phi = exp(-theta); half-life = ln 2 / theta bars.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from usdjpy_mr.config import OuConfig


@dataclass(frozen=True)
class OuFit:
    theta: float  # mean-reversion speed per bar
    mu: float
    sigma: float  # innovation std per bar
    half_life: float  # bars; inf if theta <= 0
    slope: float  # AR(1) slope on lagged level, b = -theta
    slope_se: float
    slope_t: float
    nobs: int
    ci_low: float | None = None  # bootstrap CI on half-life
    ci_high: float | None = None
    n_boot: int = 0

    def as_dict(self) -> dict:
        return {
            "theta": self.theta,
            "mu": self.mu,
            "sigma": self.sigma,
            "half_life": self.half_life,
            "slope": self.slope,
            "slope_se": self.slope_se,
            "slope_t": self.slope_t,
            "nobs": self.nobs,
            "hl_ci_low": self.ci_low,
            "hl_ci_high": self.ci_high,
            "n_boot": self.n_boot,
        }


def _ar1(e: np.ndarray) -> tuple[float, float, float, float, np.ndarray]:
    """OLS of de_t on [1, e_{t-1}]. Returns (a, b, se_b, sigma, residuals)."""
    lag = e[:-1]
    de = np.diff(e)
    X = np.column_stack([np.ones_like(lag), lag])
    coef, *_ = np.linalg.lstsq(X, de, rcond=None)
    fitted = X @ coef
    u = de - fitted
    dof = max(len(de) - 2, 1)
    s2 = float(u @ u) / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se_b = float(np.sqrt(s2 * xtx_inv[1, 1]))
    return float(coef[0]), float(coef[1]), se_b, float(np.sqrt(s2)), u


def _theta(b: float) -> float:
    """Exact mapping from the AR(1) slope on the lagged level: phi = 1 + b = exp(-theta)."""
    phi = 1.0 + b
    return float(-np.log(phi)) if 0.0 < phi < 1.0 else 0.0


def _half_life(b: float) -> float:
    theta = _theta(b)
    return float(np.log(2) / theta) if theta > 0 else float("inf")


def fit_ou(resid: pd.Series | np.ndarray, cfg: OuConfig) -> OuFit:
    e = np.asarray(pd.Series(resid).dropna(), dtype=float)
    if len(e) < 30:
        raise ValueError(f"OU fit needs >= 30 observations, got {len(e)}")
    a, b, se_b, sigma, u = _ar1(e)
    theta = _theta(b)
    mu = -a / b if b < 0 else float("nan")
    hl = _half_life(b)
    ci_low = ci_high = None
    if cfg.n_boot > 0:
        rng = np.random.default_rng(cfg.seed)
        hls = np.empty(cfg.n_boot)
        n = len(e)
        phi = 1.0 + b
        for i in range(cfg.n_boot):
            # residual bootstrap: rebuild a path from the fitted AR(1) with resampled innovations
            # sim[t] = phi * sim[t-1] + a + shock[t]  (AR(1) recursion via lfilter, x0 = e[0])
            shocks = rng.choice(u, size=n - 1, replace=True)
            drive = a + shocks
            sim = np.empty(n)
            sim[0] = e[0]
            sim[1:] = lfilter([1.0], [1.0, -phi], drive, zi=[phi * e[0]])[0]
            _, bb, *_ = _ar1(sim)
            hls[i] = _half_life(bb)
        lo, hi = (1 - cfg.ci) / 2, 1 - (1 - cfg.ci) / 2
        finite = hls[np.isfinite(hls)]
        if len(finite) >= max(10, int(0.5 * cfg.n_boot)):
            ci_low, ci_high = float(np.quantile(finite, lo)), float(np.quantile(finite, hi))
            if len(finite) < cfg.n_boot:  # some draws had no mean reversion: upper bound is open
                ci_high = float("inf")
        else:
            ci_low, ci_high = float("nan"), float("inf")
    return OuFit(
        theta=theta,
        mu=mu,
        sigma=sigma,
        half_life=hl,
        slope=b,
        slope_se=se_b,
        slope_t=b / se_b if se_b > 0 else float("nan"),
        nobs=len(e),
        ci_low=ci_low,
        ci_high=ci_high,
        n_boot=cfg.n_boot,
    )


def simulate_ou(
    theta: float, mu: float, sigma: float, n: int, seed: int = 0, x0: float | None = None
) -> np.ndarray:
    """Exact discretisation of an OU process at unit step; used by tests and bootstraps."""
    rng = np.random.default_rng(seed)
    phi = np.exp(-theta)
    sd = sigma * np.sqrt((1 - phi**2) / (2 * theta)) if theta > 0 else sigma
    out = np.empty(n)
    out[0] = mu if x0 is None else x0
    z = rng.standard_normal(n)
    for t in range(1, n):
        out[t] = mu + phi * (out[t - 1] - mu) + sd * z[t]
    return out
