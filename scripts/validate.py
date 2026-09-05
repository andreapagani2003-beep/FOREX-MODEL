#!/usr/bin/env python
"""Validate data/processed/daily.parquet. Exit code 1 on any error.

Usage:
    uv run scripts/validate.py [--config configs/default.yaml] [--root .]
"""

from __future__ import annotations

import argparse
import sys

from usdjpy_mr.config import load_config
from usdjpy_mr.data.pipeline import load_daily
from usdjpy_mr.data.validate import validate_daily
from usdjpy_mr.utils.logging import get_logger

log = get_logger("scripts.validate")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--root", default=None)
    args = p.parse_args(argv)

    cfg = load_config(args.config, args.root)
    df, meta = load_daily(cfg)
    res = validate_daily(df, cfg, meta)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.reports_dir / "phase1_validation.md"
    out.write_text(res.to_markdown(), encoding="utf-8")
    print(res.to_markdown())
    print(f"report written to {out}")
    return 0 if res.ok else 1


if __name__ == "__main__":
    sys.exit(main())
