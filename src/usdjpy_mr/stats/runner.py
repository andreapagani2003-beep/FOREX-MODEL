"""Phase 2 driver: run every test for every (specification, sample) and collect results."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from usdjpy_mr.config import Config, SampleWindow, SpecConfig, StatsConfig
from usdjpy_mr.stats.beta import KalmanResult, kalman_beta, rolling_ols, static_ols, zscore
from usdjpy_mr.stats.breaks import chow_test, explain_breaks, sup_f_breaks
from usdjpy_mr.stats.cointegration import engle_granger_both, johansen, rolling_engle_granger
from usdjpy_mr.stats.ou import fit_ou
from usdjpy_mr.stats.stationarity import integration_order
from usdjpy_mr.utils.logging import get_logger

log = get_logger(__name__)


def spec_series(df: pd.DataFrame, spec: SpecConfig) -> tuple[pd.Series, pd.Series]:
    """(y, x) for a specification. y = usdjpy; x = the chosen spread."""
    y = df["usdjpy"].rename("usdjpy")
    if spec.yield_type == "nominal":
        x = df[f"spread{spec.tenor}"].rename(f"spread{spec.tenor}")
    else:
        if spec.tenor != "10y" or "us10y_real" not in df.columns:
            raise ValueError("real-yield spec needs tenor 10y and column us10y_real")
        # US real (TIPS) minus nominal JGB: no JGB real-yield source; documented limitation.
        x = (df["us10y_real"] - df["jgb10y"]).rename("spread10y_real_us")
    return y, x


def slice_sample(df: pd.DataFrame, window: SampleWindow) -> pd.DataFrame:
    start = pd.Timestamp(window.start) if window.start else None
    end = pd.Timestamp(window.end) if window.end else None
    return df.loc[start:end]


@dataclass
class SpecSampleResult:
    spec: str
    sample: str
    start: str
    end: str
    nobs: int
    tables: dict = field(default_factory=dict)  # JSON-able numbers
    series: dict[str, pd.Series | pd.DataFrame] = field(default_factory=dict)  # for plots


def _anchor_checks(resid: pd.Series, z: pd.Series, cfg: StatsConfig) -> list[dict]:
    rows = []
    for a in cfg.anchors:
        sl = slice(pd.Timestamp(a.start), pd.Timestamp(a.end))
        r, zz = resid.loc[sl], z.loc[sl]
        if len(r) == 0:
            rows.append(
                {
                    "window": f"{a.start}..{a.end}",
                    "expected": a.expected_sign,
                    "in_sample": False,
                    "pass": None,
                }
            )
            continue
        sign = "negative" if r.mean() < 0 else "positive"
        rows.append(
            {
                "window": f"{a.start}..{a.end}",
                "expected": a.expected_sign,
                "in_sample": True,
                "resid_mean": float(r.mean()),
                "z_mean": float(zz.mean()) if zz.notna().any() else None,
                "z_min": float(zz.min()) if zz.notna().any() else None,
                "z_max": float(zz.max()) if zz.notna().any() else None,
                "observed": sign,
                "pass": sign == a.expected_sign,
            }
        )
    return rows


def run_spec_sample(
    y: pd.Series,
    x: pd.Series,
    spec: SpecConfig,
    sample: str,
    cfg: StatsConfig,
    full_kalman: KalmanResult | None,
) -> SpecSampleResult:
    alpha = cfg.significance
    df = pd.concat([y, x], axis=1).dropna()
    y, x = df.iloc[:, 0], df.iloc[:, 1]
    res = SpecSampleResult(
        spec.name, sample, str(df.index[0].date()), str(df.index[-1].date()), len(df)
    )
    T = res.tables

    # 1. stationarity of levels
    T["stationarity"] = {
        "usdjpy": integration_order(y, cfg.adf, cfg.kpss, alpha),
        "spread": integration_order(x, cfg.adf, cfg.kpss, alpha),
    }
    # 2. cointegration
    eg = engle_granger_both(y, x, cfg.engle_granger)
    T["engle_granger"] = {k: v.as_dict() for k, v in eg.items()}
    T["johansen"] = johansen(y, x, cfg.johansen).as_dict()
    # 3. beta: static, rolling, kalman (kalman on the sample itself; the full-sample path is
    #    also carried for the plots so subsample starts do not re-initialise the filter)
    ols = static_ols(y, x)
    T["static_ols"] = ols.as_dict()
    roll = {w: rolling_ols(y, x, w) for w in cfg.rolling_ols_windows if w < len(df)}
    T["rolling_ols"] = {
        str(w): {
            "beta_min": float(r["beta"].min()),
            "beta_max": float(r["beta"].max()),
            "beta_last": float(r["beta"].iloc[-1]),
            "beta_std": float(r["beta"].std()),
        }
        for w, r in roll.items()
    }
    kal = kalman_beta(y, x, cfg.kalman)
    T["kalman"] = {
        "beta_last": float(kal.beta.iloc[-1]),
        "beta_min": float(kal.beta.min()),
        "beta_max": float(kal.beta.max()),
        "alpha_last": float(kal.alpha.iloc[-1]),
        "obs_var": kal.obs_var,
        "delta": kal.delta,
    }
    # 4. OU on the static residual and on the Kalman filtered residual
    T["ou_static"] = fit_ou(ols.resid, cfg.ou).as_dict()
    T["ou_kalman"] = fit_ou(kal.resid, cfg.ou).as_dict()
    # 5. breaks: known dates inside the sample (Chow), unknown (sequential sup-F)
    chow = []
    for k in cfg.breaks.known_dates:
        d = pd.Timestamp(k.date)
        if df.index[0] + pd.Timedelta(days=120) < d < df.index[-1] - pd.Timedelta(days=120):
            chow.append(
                chow_test(
                    y, x, d, k.label, n_boot=cfg.breaks.n_boot, seed=cfg.breaks.seed
                ).as_dict()
            )
    T["chow"] = chow
    supf = sup_f_breaks(y, x, cfg.breaks, alpha)
    T["sup_f"] = explain_breaks(supf.breaks, cfg.breaks, df.index[-1])
    # 7. anchors on the static residual and on the Kalman residual z-score
    z_static = zscore(ols.resid, cfg.zscore_window)
    z_kal = zscore(kal.resid, cfg.zscore_window)
    T["anchors_static"] = _anchor_checks(ols.resid, z_static, cfg)
    T["anchors_kalman"] = _anchor_checks(kal.resid, z_kal, cfg)

    # acceptance flags (handoff §5)
    lo, hi = cfg.half_life_bounds
    hl = T["ou_static"]["half_life"]
    T["acceptance"] = {
        "cointegrated_5pct": bool(
            eg["y_on_x"].pvalue < alpha or T["johansen"]["rank_trace_5pct"] >= 1
        ),
        "eg_y_on_x_p": eg["y_on_x"].pvalue,
        "johansen_rank": T["johansen"]["rank_trace_5pct"],
        "half_life_in_band": bool(lo <= hl <= hi),
        "half_life": hl,
        "unexplained_recent_break": bool(any(b["unexplained_recent"] for b in T["sup_f"])),
        "anchors_pass": all(a["pass"] for a in T["anchors_static"] if a["in_sample"])
        if any(a["in_sample"] for a in T["anchors_static"])
        else None,
    }
    T["acceptance"]["pass"] = (
        T["acceptance"]["cointegrated_5pct"]
        and T["acceptance"]["half_life_in_band"]
        and not T["acceptance"]["unexplained_recent_break"]
    )

    res.series = {
        "resid_static": ols.resid,
        "z_static": z_static,
        "kalman_beta": kal.beta,
        "kalman_alpha": kal.alpha,
        "resid_kalman": kal.resid,
        "z_kalman": z_kal,
        "rolling_beta": pd.DataFrame({f"ols_{w}": r["beta"] for w, r in roll.items()}),
        "sup_f_path": supf.f_path,
    }
    return res


def run_phase2(
    df: pd.DataFrame, cfg: Config
) -> tuple[list[SpecSampleResult], dict[str, pd.DataFrame]]:
    """All specs x samples, plus a rolling Engle-Granger series per spec on the full sample."""
    scfg = cfg.stats
    assert scfg is not None, "config has no stats section"
    results: list[SpecSampleResult] = []
    rolling_eg: dict[str, pd.DataFrame] = {}
    for spec in scfg.specs:
        y_full, x_full = spec_series(df, spec)
        log.info(
            "spec %s: rolling Engle-Granger (window %d, step %d)",
            spec.name,
            scfg.rolling_coint.window,
            scfg.rolling_coint.step,
        )
        rolling_eg[spec.name] = rolling_engle_granger(
            y_full, x_full, scfg.engle_granger, scfg.rolling_coint
        )
        for sample, window in scfg.samples.items():
            sub = slice_sample(pd.concat([y_full, x_full], axis=1), window)
            log.info(
                "spec %s / sample %s: %d obs %s..%s",
                spec.name,
                sample,
                len(sub),
                sub.index[0].date(),
                sub.index[-1].date(),
            )
            r = run_spec_sample(sub.iloc[:, 0], sub.iloc[:, 1], spec, sample, scfg, None)
            acc = r.tables["acceptance"]
            log.info(
                "  EG p=%.4f Johansen r=%d HL=%.1f [%s, %s] breaks=%d unexpl_recent=%s pass=%s",
                acc["eg_y_on_x_p"],
                acc["johansen_rank"],
                acc["half_life"],
                _fmt(r.tables["ou_static"]["hl_ci_low"]),
                _fmt(r.tables["ou_static"]["hl_ci_high"]),
                len(r.tables["sup_f"]),
                acc["unexplained_recent_break"],
                acc["pass"],
            )
            results.append(r)
    return results, rolling_eg


def _fmt(v: float | None) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "nan"
    return "inf" if v == float("inf") else f"{v:.1f}"


def summary_table(results: list[SpecSampleResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        T = r.tables
        eg, jo, acc = T["engle_granger"], T["johansen"], T["acceptance"]
        rows.append(
            {
                "spec": r.spec,
                "sample": r.sample,
                "start": r.start,
                "end": r.end,
                "nobs": r.nobs,
                "usdjpy_order": T["stationarity"]["usdjpy"]["order"],
                "spread_order": T["stationarity"]["spread"]["order"],
                "eg_p_y_on_x": eg["y_on_x"]["pvalue"],
                "eg_p_x_on_y": eg["x_on_y"]["pvalue"],
                "johansen_trace_r0": jo["trace_r0"],
                "johansen_crit95": jo["trace_r0_crit95"],
                "johansen_rank": jo["rank_trace_5pct"],
                "beta_static": T["static_ols"]["beta"],
                "beta_se": T["static_ols"]["beta_se"],
                "r2": T["static_ols"]["r2"],
                "beta_kalman_last": T["kalman"]["beta_last"],
                "beta_roll500_range": (
                    f"{T['rolling_ols']['500']['beta_min']:.1f}..{T['rolling_ols']['500']['beta_max']:.1f}"
                    if "500" in T["rolling_ols"]
                    else ""
                ),
                "half_life": T["ou_static"]["half_life"],
                "hl_ci_low": T["ou_static"]["hl_ci_low"],
                "hl_ci_high": T["ou_static"]["hl_ci_high"],
                "hl_kalman": T["ou_kalman"]["half_life"],
                "breaks_found": len(T["sup_f"]),
                "breaks_unexplained_recent": int(sum(b["unexplained_recent"] for b in T["sup_f"])),
                "anchors_pass": acc["anchors_pass"],
                "cointegrated_5pct": acc["cointegrated_5pct"],
                "hl_in_band": acc["half_life_in_band"],
                "pass": acc["pass"],
            }
        )
    return pd.DataFrame(rows)


def recommend(summary: pd.DataFrame, gate_sample: str = "post_2016") -> dict:
    """Primary/secondary spec: passing on the gate sample, ranked by EG p-value then half-life."""
    gate = summary[(summary["sample"] == gate_sample) & summary["pass"]].copy()
    if gate.empty:
        return {
            "primary": None,
            "secondary": None,
            "reason": f"no specification passes on {gate_sample}",
        }
    gate = gate.sort_values(["eg_p_y_on_x", "half_life"])
    specs = gate["spec"].tolist()
    return {
        "primary": specs[0],
        "secondary": specs[1] if len(specs) > 1 else None,
        "reason": f"EG p-value then half-life, among specs passing on {gate_sample}",
    }


def today() -> dt.date:
    return dt.date.today()
