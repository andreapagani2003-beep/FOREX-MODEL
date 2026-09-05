"""Runner + report smoke test on a synthetic cointegrated dataset (fast bootstrap settings)."""

from pathlib import Path

import numpy as np
import pandas as pd
from tests.conftest import REPO_ROOT, make_config

from usdjpy_mr.stats.ou import simulate_ou
from usdjpy_mr.stats.report import dump_results, spec_figure, write_report
from usdjpy_mr.stats.runner import recommend, run_phase2, summary_table


def _synthetic_daily(n: int = 1500, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    idx.name = "date"
    us10 = 2.0 + np.cumsum(rng.normal(0, 0.02, n))
    jgb10 = 0.3 + np.cumsum(rng.normal(0, 0.005, n))
    us2 = us10 - 0.5 + np.cumsum(rng.normal(0, 0.01, n))
    jgb2 = jgb10 - 0.3
    spread10 = us10 - jgb10
    e = simulate_ou(np.log(2) / 20, 0, 1.5, n, seed=seed + 1)
    df = pd.DataFrame(
        {
            "usdjpy": 100 + 15 * spread10 + e,
            "us2y": us2,
            "us10y": us10,
            "jgb2y": jgb2,
            "jgb10y": jgb10,
            "us10y_real": us10 - 2.0,
        },
        index=idx,
    )
    df["spread2y"] = df["us2y"] - df["jgb2y"]
    df["spread10y"] = df["us10y"] - df["jgb10y"]
    return df


def test_run_phase2_and_report(tmp_path: Path):
    # copy configs so the tmp root is a valid project root
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "default.yaml").write_text(
        (REPO_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    cfg = make_config(
        tmp_path,
        stats={
            "ou": {"n_boot": 50},
            "breaks": {"n_boot": 19},
            "rolling_coint": {"window": 300, "step": 100},
            "samples": {
                "full": {"start": None, "end": None},
                "post_2016": {"start": "2021-01-01", "end": None},
            },
            "anchors": [{"start": "2022-01-03", "end": "2022-01-31", "expected_sign": "negative"}],
        },
    )
    df = _synthetic_daily()
    results, rolling_eg = run_phase2(df, cfg)
    assert len(results) == len(cfg.stats.specs) * len(cfg.stats.samples)
    s = summary_table(results)
    ten = s[(s["spec"] == "10y_nominal") & (s["sample"] == "full")].iloc[0]
    assert ten["cointegrated_5pct"]
    assert ten["hl_in_band"]
    assert 10 < ten["half_life"] < 40
    assert 12 < ten["beta_static"] < 18
    rec = recommend(s)
    assert rec["primary"] is not None

    figs = {
        r.spec: spec_figure(
            r, rolling_eg[r.spec], df, cfg, cfg.reports_dir / "figures" / f"{r.spec}.png"
        )
        for r in results
        if r.sample == "full"
    }
    assert all(p.exists() and p.stat().st_size > 10_000 for p in figs.values())
    out = write_report(results, rolling_eg, df, cfg, figs)
    text = out.read_text(encoding="utf-8")
    assert "Acceptance MET" in text
    assert "| 10y_nominal | full |" in text
    d = dump_results(results, rolling_eg, cfg)
    assert (d / "results.json").exists()
