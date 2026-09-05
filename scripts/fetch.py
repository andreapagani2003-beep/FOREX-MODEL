#!/usr/bin/env python
"""Fetch raw sources and build data/processed/daily.parquet + metadata.json.

Usage:
    uv run scripts/fetch.py [--config configs/default.yaml] [--root .] [--fx-provider yahoo|ibkr]
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from usdjpy_mr.config import load_config
from usdjpy_mr.data.pipeline import build_daily, fetch_all
from usdjpy_mr.utils.logging import get_logger

log = get_logger("scripts.fetch")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None, help="YAML config (default configs/default.yaml)")
    p.add_argument("--root", default=None, help="project root (default $USDJPY_MR_ROOT or cwd)")
    p.add_argument(
        "--fx-provider",
        choices=["yahoo", "ibkr"],
        default=None,
        help="override sources.fx.provider",
    )
    args = p.parse_args(argv)

    cfg = load_config(args.config, args.root)
    load_dotenv(cfg.root / ".env")
    sources, raw_files = fetch_all(cfg, fx_provider=args.fx_provider)
    df, meta = build_daily(cfg, sources, raw_files)
    print(
        f"rows={len(df)} first={meta['first_date']} last={meta['last_date']} "
        f"convention={meta['alignment']['convention']} dropped={meta['alignment']['rows_dropped']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
