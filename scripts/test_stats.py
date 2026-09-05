#!/usr/bin/env python
"""Phase 2: run every statistical test on data/processed/daily.parquet and write the report.

Usage:
    uv run scripts/test_stats.py [--config configs/default.yaml] [--root .]
Outputs: reports/phase2_summary.md, reports/phase2/results.json, reports/figures/phase2_<spec>.png
"""

from __future__ import annotations

import argparse
import sys
import warnings

from usdjpy_mr.config import load_config
from usdjpy_mr.data.pipeline import load_daily
from usdjpy_mr.stats.report import dump_results, spec_figure, write_report
from usdjpy_mr.stats.runner import run_phase2, summary_table
from usdjpy_mr.utils.logging import get_logger

log = get_logger("scripts.test_stats")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--root", default=None)
    args = p.parse_args(argv)
    cfg = load_config(args.config, args.root)
    df, _ = load_daily(cfg)
    warnings.simplefilter("ignore", FutureWarning)
    results, rolling_eg = run_phase2(df, cfg)
    figures = {}
    for r in results:
        if r.sample == "full":
            figures[r.spec] = spec_figure(
                r, rolling_eg[r.spec], df, cfg, cfg.reports_dir / "figures" / f"phase2_{r.spec}.png"
            )
    dump_results(results, rolling_eg, cfg)
    out = write_report(results, rolling_eg, df, cfg, figures)
    s = summary_table(results)
    print(
        s[
            [
                "spec",
                "sample",
                "eg_p_y_on_x",
                "johansen_rank",
                "half_life",
                "breaks_unexplained_recent",
                "pass",
            ]
        ].to_string(index=False, float_format=lambda v: f"{v:.3g}")
    )
    print(f"report: {out}")
    return 0 if bool(s["pass"].any()) else 2


if __name__ == "__main__":
    sys.exit(main())
