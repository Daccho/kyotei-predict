"""Filesystem layout for the project.

Raw archives live under data/raw/ and are never deleted (re-runs skip them).
Extracted fixed-width payloads live under data/extracted/ so parsing is
repeatable without touching the archives again.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("KYOTEI_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
EXTRACTED_DIR = DATA_DIR / "extracted"
PARQUET_DIR = DATA_DIR / "parquet"
REPORTS_DIR = Path(os.environ.get("KYOTEI_REPORTS_DIR", PROJECT_ROOT / "reports"))

# Ledger of request outcomes, so known-404 (non-race) days are not re-requested.
LEDGER_PATH = RAW_DIR / "_ledger.json"

#: The two daily feed kinds. B = 番組表 (race card), K = 競走成績 (results).
DAILY_KINDS = ("B", "K")


def raw_path(kind: str, day: date) -> Path:
    """Local archive path mirroring the upstream directory layout.

    >>> raw_path("B", date(2020, 7, 14)).as_posix().endswith("data/raw/B/202007/b200714.lzh")
    True
    """
    _check_kind(kind)
    return RAW_DIR / kind / f"{day:%Y%m}" / f"{kind.lower()}{day:%y%m%d}.lzh"


def extracted_path(kind: str, day: date) -> Path:
    """Local path for the decompressed fixed-width payload."""
    _check_kind(kind)
    return EXTRACTED_DIR / kind / f"{day:%Y%m}" / f"{kind.lower()}{day:%y%m%d}.txt"


def fan_raw_path(stem: str) -> Path:
    """Local archive path for a half-yearly racer file, e.g. stem='fan1410'."""
    return RAW_DIR / "fan" / f"{stem}.lzh"


def fan_extracted_path(stem: str) -> Path:
    return EXTRACTED_DIR / "fan" / f"{stem}.txt"


def _check_kind(kind: str) -> None:
    if kind not in DAILY_KINDS:
        raise ValueError(f"kind must be one of {DAILY_KINDS}, got {kind!r}")
