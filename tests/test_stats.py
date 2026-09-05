"""Statistical functions tested against synthetic series with known properties."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tests.conftest import make_config

from usdjpy_mr.stats.beta import kalman_beta, rolling_ols, static_ols, zscore
from usdjpy_mr.stats.breaks import chow_test, explain_breaks, sup_f_breaks
from usdjpy_mr.stats.cointegration import (
    engle_granger,
    engle_granger_both,
    johansen,
    rolling_engle_granger,
)
from usdjpy_mr.stats.ou import fit_ou, simulate_ou
from usdjpy_mr.stats.stationarity import adf_test, integration_order, kpss_test

N = 2000
IDX = pd.bdate_range("2012-01-02", periods=N)


@pytest.fixture(scope="module")
def stats_cfg():
    return make_config(Path("."), stats={"ou": {"n_boot": 200}, "breaks": {"n_boot": 99}}).stats


def _random_walk(seed: int, n: int = N, sd: float = 1.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(np.cumsum(rng.normal(0, sd, n)), index=IDX[:n], name="x")


def _cointegrated_pair(
    seed: int, alpha: float = 100.0, beta: float = 12.0, half_life: float = 20.0
):
    x = _random_walk(seed, sd=0.1)
    theta = np.log(2) / half_life
    e = simulate_ou(theta=theta, mu=0.0, sigma=1.0, n=N, seed=seed + 1)
    y = pd.Series(alpha + beta * x.to_numpy() + e, index=IDX, name="y")
    return y, x, e


# --- stationarity -----------------------------------------------------------------------------
def test_adf_kpss_on_random_walk_and_ou(stats_cfg):
    rw = _random_walk(1)
    ou = pd.Series(simulate_ou(np.log(2) / 10, 0, 1, N, seed=2), index=IDX)
    assert adf_test(rw, stats_cfg.adf).pvalue > 0.05
    assert kpss_test(rw, stats_cfg.kpss).pvalue < 0.05
    assert adf_test(ou, stats_cfg.adf).pvalue < 0.01
    assert kpss_test(ou, stats_cfg.kpss).pvalue > 0.05


def test_integration_order(stats_cfg):
    assert (
        integration_order(_random_walk(3), stats_cfg.adf, stats_cfg.kpss, 0.05)["order"] == "I(1)"
    )
    ou = pd.Series(simulate_ou(np.log(2) / 10, 0, 1, N, seed=4), index=IDX)
    assert integration_order(ou, stats_cfg.adf, stats_cfg.kpss, 0.05)["order"] == "I(0)"


# --- cointegration ----------------------------------------------------------------------------
def test_engle_granger_detects_cointegration(stats_cfg):
    y, x, _ = _cointegrated_pair(10)
    r = engle_granger(y, x, stats_cfg.engle_granger)
    assert r.pvalue < 0.01
    assert r.beta == pytest.approx(12.0, abs=0.5)
    assert r.alpha == pytest.approx(100.0, abs=3.0)
    both = engle_granger_both(y, x, stats_cfg.engle_granger)
    assert both["x_on_y"].pvalue < 0.01


def test_engle_granger_independent_walks_not_cointegrated(stats_cfg):
    y, x = _random_walk(20).rename("y"), _random_walk(21)
    assert engle_granger(y, x, stats_cfg.engle_granger).pvalue > 0.05


def test_johansen_rank(stats_cfg):
    y, x, _ = _cointegrated_pair(30)
    r = johansen(y, x, stats_cfg.johansen)
    assert r.rank_trace_5pct == 1
    assert r.rank_maxeig_5pct == 1
    assert -r.cointegrating_vector[1] == pytest.approx(12.0, abs=0.6)
    y2, x2 = _random_walk(31).rename("y"), _random_walk(32)
    assert johansen(y2, x2, stats_cfg.johansen).rank_trace_5pct == 0


def test_rolling_engle_granger_shape_and_no_lookahead(stats_cfg):
    y, x, _ = _cointegrated_pair(40)
    cfg = stats_cfg.rolling_coint.model_copy(update={"window": 500, "step": 50})
    r = rolling_engle_granger(y, x, stats_cfg.engle_granger, cfg)
    assert r.index[0] == IDX[499]
    assert r["reject_5pct"].mean() > 0.6
    # poison the future: statistics up to a date must not change
    y2 = y.copy()
    y2.iloc[1500:] += 1000.0
    r2 = rolling_engle_granger(y2, x, stats_cfg.engle_granger, cfg)
    cut = IDX[1499]
    pd.testing.assert_frame_equal(r.loc[:cut], r2.loc[:cut])


# --- beta -------------------------------------------------------------------------------------
def test_static_and_rolling_ols(stats_cfg):
    y, x, _ = _cointegrated_pair(50)
    s = static_ols(y, x)
    assert s.beta == pytest.approx(12.0, abs=0.5)
    r = rolling_ols(y, x, 250)
    assert r["beta"].isna().sum() == 249
    assert r["beta"].iloc[-1] == pytest.approx(s.beta, abs=1.5)
    # exactness: last window equals a direct OLS on the same rows
    direct = static_ols(y.iloc[-250:], x.iloc[-250:])
    assert r["beta"].iloc[-1] == pytest.approx(direct.beta, rel=1e-6)
    assert r["alpha"].iloc[-1] == pytest.approx(direct.alpha, rel=1e-6)


def test_rolling_ols_no_lookahead():
    y, x, _ = _cointegrated_pair(51)
    a = rolling_ols(y, x, 250)
    y2 = y.copy()
    y2.iloc[1000:] += 50.0
    b = rolling_ols(y2, x, 250)
    pd.testing.assert_frame_equal(a.iloc[:1000], b.iloc[:1000])


def test_kalman_tracks_constant_and_step_beta(stats_cfg):
    y, x, _ = _cointegrated_pair(60)
    k = kalman_beta(y, x, stats_cfg.kalman)
    assert k.beta.iloc[-500:].mean() == pytest.approx(12.0, abs=0.7)
    # step change in beta halfway; a looser delta lets the filter move
    x2 = _random_walk(61, sd=0.1)
    beta_path = np.where(np.arange(N) < N // 2, 12.0, 6.0)
    e = simulate_ou(np.log(2) / 20, 0, 1, N, seed=62)
    y2 = pd.Series(100 + beta_path * x2.to_numpy() + e, index=IDX, name="y")
    k2 = kalman_beta(y2, x2, stats_cfg.kalman.model_copy(update={"delta": 1e-3}))
    assert k2.beta.iloc[N // 2 - 200 : N // 2].mean() == pytest.approx(12.0, abs=1.5)
    assert k2.beta.iloc[-200:].mean() == pytest.approx(6.0, abs=1.5)
    # innovation uses the prior state: poisoning y at t must not change innovations before t
    y3 = y.copy()
    y3.iloc[1500:] += 100.0
    k3 = kalman_beta(y3, x, stats_cfg.kalman)
    pd.testing.assert_series_equal(k.innovation.iloc[:1500], k3.innovation.iloc[:1500])


def test_zscore_window():
    e = pd.Series(np.arange(100, dtype=float), index=IDX[:100])
    z = zscore(e, 10)
    assert z.isna().sum() == 9
    assert z.iloc[-1] == pytest.approx(
        (99 - np.mean(range(90, 100))) / np.std(range(90, 100), ddof=1)
    )


# --- OU ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("hl", [5.0, 20.0, 40.0])
def test_ou_half_life_recovered(stats_cfg, hl):
    e = simulate_ou(theta=np.log(2) / hl, mu=0.0, sigma=1.0, n=4000, seed=int(hl))
    fit = fit_ou(e, stats_cfg.ou)
    assert fit.half_life == pytest.approx(hl, rel=0.25)
    assert fit.ci_low <= fit.half_life <= fit.ci_high
    assert fit.ci_low < hl < fit.ci_high
    assert fit.slope_t < -3


def test_ou_random_walk_has_no_half_life(stats_cfg):
    fit = fit_ou(_random_walk(70).to_numpy(), stats_cfg.ou.model_copy(update={"n_boot": 0}))
    assert fit.half_life > 100 or not np.isfinite(fit.half_life)
    assert fit.slope_t > -2.86  # no significant mean reversion (ADF 5% critical value)


# --- breaks -----------------------------------------------------------------------------------
def _pair_with_break(seed: int, k: int, beta_after: float):
    x = _random_walk(seed, sd=0.1)
    beta_path = np.where(np.arange(N) < k, 12.0, beta_after)
    e = simulate_ou(np.log(2) / 20, 0, 1, N, seed=seed + 1)
    y = pd.Series(100 + beta_path * x.to_numpy() + e, index=IDX, name="y")
    return y, x


def test_chow_known_break():
    y, x = _pair_with_break(80, 1200, 4.0)
    r = chow_test(y, x, IDX[1200], "synthetic", n_boot=99, seed=1)
    assert r.pvalue < 1e-6
    assert r.pvalue_boot <= 0.02
    assert r.beta_before == pytest.approx(12.0, abs=1.5)
    assert r.beta_after == pytest.approx(4.0, abs=1.5)
    y0, x0, _ = _cointegrated_pair(81)
    r0 = chow_test(y0, x0, IDX[1200], n_boot=99, seed=2)
    assert r0.pvalue_boot > 0.05


def test_sup_f_finds_break_and_is_quiet_without_one(stats_cfg):
    y, x = _pair_with_break(90, 1300, 4.0)
    r = sup_f_breaks(y, x, stats_cfg.breaks, 0.05)
    assert len(r.breaks) == 1
    assert abs(IDX.get_loc(pd.Timestamp(r.breaks[0].date)) - 1300) < 40
    assert r.breaks[0].pvalue < 0.05
    y0, x0, _ = _cointegrated_pair(91)
    r0 = sup_f_breaks(y0, x0, stats_cfg.breaks, 0.05)
    assert len(r0.breaks) == 0


def test_explain_breaks(stats_cfg):
    y, x = _pair_with_break(92, 1300, 4.0)
    r = sup_f_breaks(y, x, stats_cfg.breaks, 0.05)
    rows = explain_breaks(r.breaks, stats_cfg.breaks, IDX[-1])
    assert rows and {"nearest_known", "explained", "unexplained_recent"} <= set(rows[0])
