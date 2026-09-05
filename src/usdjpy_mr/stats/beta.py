"""Beta estimation: static OLS, rolling OLS, and a Kalman filter with time-varying (alpha, beta).

No-look-ahead contract: for every row t, `rolling_ols` and `kalman_beta` return estimates that
use observations up to and including t (the *filtered* estimate). Consumers that need a signal
for t computed from t-1 information must shift by one bar; that shift lives in the signal code,
not here, so the estimation functions stay pure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from usdjpy_mr.config import KalmanConfig


@dataclass(frozen=True)
class OlsResult:
    alpha: float
    beta: float
    alpha_se: float
    beta_se: float
    r2: float
    nobs: int
    resid: pd.Series

    def as_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "alpha_se": self.alpha_se,
            "beta_se": self.beta_se,
            "r2": self.r2,
            "nobs": self.nobs,
        }


def static_ols(y: pd.Series, x: pd.Series) -> OlsResult:
    df = pd.concat([y, x], axis=1).dropna()
    X = sm.add_constant(df.iloc[:, 1].to_numpy())
    fit = sm.OLS(df.iloc[:, 0].to_numpy(), X).fit()
    resid = pd.Series(fit.resid, index=df.index, name="resid")
    return OlsResult(
        float(fit.params[0]),
        float(fit.params[1]),
        float(fit.bse[0]),
        float(fit.bse[1]),
        float(fit.rsquared),
        int(fit.nobs),
        resid,
    )


def rolling_ols(y: pd.Series, x: pd.Series, window: int) -> pd.DataFrame:
    """Rolling OLS y = a + b x over the trailing `window` rows ending at t (inclusive).
    Returns DataFrame[alpha, beta, resid] with NaN for the first window-1 rows. Vectorised via
    rolling moments, so it is exact OLS without a per-window fit."""
    df = pd.concat([y, x], axis=1).dropna()
    yv, xv = df.iloc[:, 0], df.iloc[:, 1]
    mx, my = xv.rolling(window).mean(), yv.rolling(window).mean()
    cov = (xv * yv).rolling(window).mean() - mx * my
    var = (xv * xv).rolling(window).mean() - mx * mx
    beta = cov / var
    alpha = my - beta * mx
    resid = yv - (alpha + beta * xv)
    return pd.DataFrame({"alpha": alpha, "beta": beta, "resid": resid}, index=df.index)


@dataclass
class KalmanResult:
    alpha: pd.Series
    beta: pd.Series
    resid: pd.Series  # y_t - (alpha_t|t + beta_t|t x_t): filtered residual
    innovation: pd.Series  # y_t - (alpha_t|t-1 + beta_t|t-1 x_t): one-step-ahead prediction error
    innovation_var: pd.Series
    obs_var: float
    delta: float


def kalman_beta(y: pd.Series, x: pd.Series, cfg: KalmanConfig) -> KalmanResult:
    """Time-varying regression y_t = alpha_t + beta_t x_t + e_t with random-walk states.

    State s_t = [alpha_t, beta_t], s_t = s_{t-1} + w_t, w ~ N(0, Q), Q = delta/(1-delta) * I.
    Observation y_t = H_t s_t + v_t, H_t = [1, x_t], v ~ N(0, R). R defaults to the static-OLS
    residual variance on the first `init_from_first_n` observations, which also initialise the
    state; the initial covariance is that OLS parameter covariance.
    """
    df = pd.concat([y, x], axis=1).dropna()
    yv, xv = df.iloc[:, 0].to_numpy(dtype=float), df.iloc[:, 1].to_numpy(dtype=float)
    n = len(df)
    n0 = min(cfg.init_from_first_n, n)
    init = sm.OLS(yv[:n0], sm.add_constant(xv[:n0])).fit()
    R = float(cfg.obs_var) if cfg.obs_var is not None else float(np.var(init.resid, ddof=2))
    Q = (cfg.delta / (1.0 - cfg.delta)) * np.eye(2)

    s = np.asarray(init.params, dtype=float)
    P = np.asarray(init.cov_params(), dtype=float)
    alpha = np.empty(n)
    beta = np.empty(n)
    resid = np.empty(n)
    innov = np.empty(n)
    innov_var = np.empty(n)
    for t in range(n):
        # predict
        P = P + Q
        H = np.array([1.0, xv[t]])
        yhat = H @ s
        F = float(H @ P @ H + R)
        e = yv[t] - yhat
        # update
        K = (P @ H) / F
        s = s + K * e
        P = P - np.outer(K, H) @ P
        alpha[t], beta[t] = s
        innov[t], innov_var[t] = e, F
        resid[t] = yv[t] - (s[0] + s[1] * xv[t])
    idx = df.index
    return KalmanResult(
        alpha=pd.Series(alpha, idx, name="alpha"),
        beta=pd.Series(beta, idx, name="beta"),
        resid=pd.Series(resid, idx, name="resid"),
        innovation=pd.Series(innov, idx, name="innovation"),
        innovation_var=pd.Series(innov_var, idx, name="innovation_var"),
        obs_var=R,
        delta=cfg.delta,
    )


def zscore(resid: pd.Series, window: int) -> pd.Series:
    """Rolling z-score of a residual using the trailing `window` rows ending at t (inclusive)."""
    m = resid.rolling(window).mean()
    s = resid.rolling(window).std(ddof=1)
    return ((resid - m) / s).rename("zscore")
