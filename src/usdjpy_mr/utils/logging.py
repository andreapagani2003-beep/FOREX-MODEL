from __future__ import annotations

import logging
import sys

_FMT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Module logger writing to stderr; configured once, idempotent."""
    root = logging.getLogger("usdjpy_mr")
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FMT, datefmt="%Y-%m-%d %H:%M:%S"))
        root.addHandler(handler)
        root.setLevel(level)
    return logging.getLogger(name if name.startswith("usdjpy_mr") else f"usdjpy_mr.{name}")
