"""Download, archive and hash raw vendor files. Raw files are never modified after download."""

from __future__ import annotations

import datetime as dt
import hashlib
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from usdjpy_mr.utils.logging import get_logger

log = get_logger(__name__)

USER_AGENT = "usdjpy-mr/0.1 (+research pipeline; python-urllib)"


@dataclass(frozen=True)
class RawFile:
    """Provenance record for one downloaded file; serialised into metadata.json."""

    source: str
    url: str
    path: str
    downloaded_at: str
    sha256: str
    bytes: int

    def as_dict(self) -> dict[str, str | int]:
        return self.__dict__.copy()


def utc_now_iso() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def http_get_bytes(url: str, timeout: float = 60.0) -> bytes:
    """GET `url` and return the body. Honors HTTPS_PROXY etc. via urllib defaults."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - https URLs from config
        status = getattr(resp, "status", 200)
        if status != 200:
            raise OSError(f"HTTP {status} for {url}")
        return resp.read()


def download_raw(source: str, url: str, dest: Path, timeout: float = 60.0) -> tuple[bytes, RawFile]:
    """Download `url` to `dest` (overwriting) and return (bytes, provenance record)."""
    log.info("downloading %s <- %s", dest.name, url)
    data = http_get_bytes(url, timeout=timeout)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    rec = RawFile(
        source=source,
        url=url,
        path=str(dest),
        downloaded_at=utc_now_iso(),
        sha256=sha256_bytes(data),
        bytes=len(data),
    )
    log.info("saved %s (%d bytes, sha256 %s)", dest, rec.bytes, rec.sha256[:12])
    return data, rec


def git_sha(root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None
