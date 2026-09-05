"""Structural breaks in y = a + b x + e.

- `chow_test`: known break date. F-test of a single regression vs two regressions split at the
  date (breaks in (a, b)), plus an F-ratio test of residual variance across the two segments.
- `sup_f_breaks`: unknown breaks. Quandt-Andrews sup-F over all candidate dates inside the
  trimmed range, with a residual-bootstrap p-value (the asymptotic distribution is non-standard).
  Applied sequentially on sub-segments up to `max_breaks`, in the spirit of Bai-Perron's
  sequential procedure. Split-sample SSRs come from prefix/suffix cumulative sums, so the whole
  candidate sweep is O(n) and the bootstrap is cheap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.signal import lfilter
from statsmodels.tsa.ar_model import AutoReg, ar_select_order

from usdjpy_mr.config import BreaksConfig

_K = 2  # regressors: intercept and slope
_MAX_AR_LAG = 10


class _SieveBootstrap:
    """Resample a residual series through a fitted AR(p), preserving its dependence structure."""

    def __init__(self, u: np.ndarray, rng: np.random.Generator):
        self.rng = rng
        self.n = len(u)
        sel = ar_select_order(u, maxlag=_MAX_AR_LAG, ic="aic", trend="n")
        lags = sel.ar_lags or []
        self.p = max(lags) if lags else 0
        if self.p:
            fit = AutoReg(u, lags=self.p, trend="n").fit()
            self.coef = np.asarray(fit.params, dtype=float)
            self.innov = np.asarray(fit.resid, dtype=float)
        else:
            self.coef = np.zeros(0)
            self.innov = u - u.mean()
        self.innov = self.innov - self.innov.mean()

    def draw(self) -> np.ndarray:
        n, p = self.n, self.p
        eps = self.rng.choice(self.innov, size=n + 100, replace=True)  # 100-step burn-in
        if p == 0:
            return eps[100:]
        out = lfilter([1.0], np.concatenate([[1.0], -self.coef]), eps)
        return out[100:]


def _fitted_and_resid(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ coef
    return fitted, y - fitted


def _ssr_prefix_suffix(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """SSR of OLS y~1+x on prefixes [0:i] (i=1..n) and suffixes [i:n] (i=0..n-1), via cumsums."""

    def _ssr_from_moments(n, sx, sy, sxx, sxy, syy):
        with np.errstate(divide="ignore", invalid="ignore"):
            var = sxx - sx * sx / n
            cov = sxy - sx * sy / n
            ssr = (syy - sy * sy / n) - np.where(var > 0, cov * cov / var, 0.0)
        return np.where(n >= _K + 1, ssr, np.nan)

    cs = lambda a: np.concatenate([[0.0], np.cumsum(a)])  # noqa: E731
    Sx, Sy, Sxx, Sxy, Syy = cs(x), cs(y), cs(x * x), cs(x * y), cs(y * y)
    n = len(x)
    i = np.arange(1, n + 1)
    pre = _ssr_from_moments(i, Sx[1:], Sy[1:], Sxx[1:], Sxy[1:], Syy[1:])
    j = np.arange(0, n)
    suf = _ssr_from_moments(
        n - j,
        Sx[n] - Sx[:-1],
        Sy[n] - Sy[:-1],
        Sxx[n] - Sxx[:-1],
        Sxy[n] - Sxy[:-1],
        Syy[n] - Syy[:-1],
    )
    return pre, suf


def _f_stats(x: np.ndarray, y: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Chow F at every candidate split i in [lo, hi): first segment [0:i], second [i:n]."""
    n = len(x)
    pre, suf = _ssr_prefix_suffix(x, y)
    ssr_full = pre[-1]
    idx = np.arange(lo, hi)
    ssr_split = pre[idx - 1] + suf[idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        f = ((ssr_full - ssr_split) / _K) / (ssr_split / (n - 2 * _K))
    return np.where(np.isfinite(f), f, -np.inf)


@dataclass(frozen=True)
class ChowResult:
    date: str
    label: str
    f_stat: float
    pvalue: (
        float  # classical F p-value (assumes iid errors; over-rejects with persistent residuals)
    )
    pvalue_boot: float  # sieve-bootstrap p-value (dependence-robust); NaN if n_boot == 0
    n_before: int
    n_after: int
    beta_before: float
    beta_after: float
    var_ratio: float  # residual variance after / before
    var_pvalue: float  # two-sided F-test on the variance ratio

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def chow_test(
    y: pd.Series,
    x: pd.Series,
    date: pd.Timestamp,
    label: str = "",
    n_boot: int = 0,
    seed: int = 0,
) -> ChowResult:
    df = pd.concat([y, x], axis=1).dropna()
    xv, yv = df.iloc[:, 1].to_numpy(dtype=float), df.iloc[:, 0].to_numpy(dtype=float)
    i = int(np.searchsorted(df.index.to_numpy(), np.datetime64(pd.Timestamp(date))))
    n = len(df)
    if i < _K + 1 or n - i < _K + 1:
        raise ValueError(f"break {date} leaves a segment too short ({i} / {n - i})")
    f = float(_f_stats(xv, yv, i, i + 1)[0])
    p = float(sps.f.sf(f, _K, n - 2 * _K))
    p_boot = float("nan")
    if n_boot > 0:
        fitted, u = _fitted_and_resid(xv, yv)
        sieve = _SieveBootstrap(u, np.random.default_rng(seed))
        exceed = sum(
            float(_f_stats(xv, fitted + sieve.draw(), i, i + 1)[0]) >= f for _ in range(n_boot)
        )
        p_boot = (exceed + 1) / (n_boot + 1)

    def _fit(xs, ys):
        X = np.column_stack([np.ones_like(xs), xs])
        coef, *_ = np.linalg.lstsq(X, ys, rcond=None)
        u = ys - X @ coef
        return float(coef[1]), float(u @ u / max(len(ys) - _K, 1))

    b0, v0 = _fit(xv[:i], yv[:i])
    b1, v1 = _fit(xv[i:], yv[i:])
    ratio = v1 / v0 if v0 > 0 else float("inf")
    d0, d1 = i - _K, n - i - _K
    cdf = sps.f.cdf(ratio, d1, d0)
    var_p = float(2 * min(cdf, 1 - cdf))
    return ChowResult(
        str(pd.Timestamp(date).date()), label, f, p, p_boot, i, n - i, b0, b1, ratio, var_p
    )


@dataclass(frozen=True)
class BreakFound:
    date: str
    sup_f: float
    pvalue: float
    segment_start: str
    segment_end: str
    beta_before: float
    beta_after: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class SupFResult:
    breaks: list[BreakFound] = field(default_factory=list)
    f_path: pd.Series | None = None  # sup-F candidate path on the full sample (for plotting)


def _sup_f_once(x: np.ndarray, y: np.ndarray, trim: float, n_boot: int, rng: np.random.Generator):
    n = len(x)
    lo, hi = max(int(np.floor(trim * n)), _K + 1), min(int(np.ceil((1 - trim) * n)), n - _K - 1)
    if hi <= lo:
        return None
    f = _f_stats(x, y, lo, hi)
    k = int(np.argmax(f))
    sup = float(f[k])
    # sieve bootstrap under H0 (single regression): y* = fitted + AR(p)-resampled residuals
    exceed = 0
    if n_boot > 0:
        fitted, u = _fitted_and_resid(x, y)
        sieve = _SieveBootstrap(u, rng)
        for _ in range(n_boot):
            if float(np.max(_f_stats(x, fitted + sieve.draw(), lo, hi))) >= sup:
                exceed += 1
    p = (exceed + 1) / (n_boot + 1) if n_boot > 0 else float("nan")
    return lo + k, sup, p, f, lo


def sup_f_breaks(y: pd.Series, x: pd.Series, cfg: BreaksConfig, alpha: float) -> SupFResult:
    """Sequential sup-F search for up to cfg.max_breaks breaks significant at `alpha`."""
    df = pd.concat([y, x], axis=1).dropna()
    xv, yv = df.iloc[:, 1].to_numpy(dtype=float), df.iloc[:, 0].to_numpy(dtype=float)
    rng = np.random.default_rng(cfg.seed)
    out = SupFResult()
    segments: list[tuple[int, int]] = [(0, len(df))]
    first = _sup_f_once(xv, yv, cfg.trim, 0, rng)
    if first is not None:
        _, _, _, f_full, lo = first
        out.f_path = pd.Series(f_full, index=df.index[lo : lo + len(f_full)], name="sup_f_path")
    found: list[tuple[int, BreakFound]] = []
    while segments and len(found) < cfg.max_breaks:
        # test the segment with the largest sup-F first
        best = None
        for s, e in segments:
            r = _sup_f_once(xv[s:e], yv[s:e], cfg.trim, cfg.n_boot, rng)
            if r is None:
                continue
            k, sup, p, _, _ = r
            if best is None or sup > best[1]:
                best = (s + k, sup, p, s, e)
        if best is None or best[2] >= alpha:
            break
        k, sup, p, s, e = best
        b0 = np.polyfit(xv[s:k], yv[s:k], 1)[0]
        b1 = np.polyfit(xv[k:e], yv[k:e], 1)[0]
        found.append(
            (
                k,
                BreakFound(
                    str(df.index[k].date()),
                    sup,
                    p,
                    str(df.index[s].date()),
                    str(df.index[e - 1].date()),
                    float(b0),
                    float(b1),
                ),
            )
        )
        segments = [seg for seg in segments if seg != (s, e)] + [(s, k), (k, e)]
        segments = [seg for seg in segments if seg[1] - seg[0] > 2 * int(cfg.trim * len(df))]
    out.breaks = [b for _, b in sorted(found, key=lambda t: t[0])]
    return out


def explain_breaks(
    found: list[BreakFound], cfg: BreaksConfig, sample_end: pd.Timestamp
) -> list[dict]:
    """Attach the nearest known policy date to each found break and flag unexplained recent ones."""
    rows = []
    recent_from = pd.Timestamp(sample_end) - pd.DateOffset(months=cfg.recent_months)
    for b in found:
        d = pd.Timestamp(b.date)
        nearest = min(cfg.known_dates, key=lambda k: abs((pd.Timestamp(k.date) - d).days))
        dist = int(abs((pd.Timestamp(nearest.date) - d).days))
        explained = dist <= cfg.tolerance_days
        rows.append(
            {
                **b.as_dict(),
                "nearest_known": str(nearest.date),
                "nearest_label": nearest.label,
                "distance_days": dist,
                "explained": explained,
                "recent": d >= recent_from,
                "unexplained_recent": (d >= recent_from) and not explained,
            }
        )
    return rows
