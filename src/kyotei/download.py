"""Download and decompress the public boatrace feeds.

Sources (see SPEC.md §3):
  番組表    https://www1.mbrace.or.jp/od2/B/{YYYYMM}/b{YYMMDD}.lzh
  競走成績  https://www1.mbrace.or.jp/od2/K/{YYYYMM}/k{YYMMDD}.lzh
  期別成績  https://www.boatrace.jp/static_extra/pc_static/download/data/kibetsu/fan{YYMM}.lzh

Rules enforced here:
  * ~1s sleep between requests. Never hammer the origin.
  * Idempotent: an already-downloaded archive is skipped; a day already known
    to be 404 (no racing) is skipped via the ledger instead of re-requested.
  * 404 is a normal outcome for non-race days and is recorded, not raised.
  * Any other failure is recorded with its status code -- never swallowed.

Decompression uses the pure-Python ``lhafile`` rather than a system ``lhasa``
binary, so the pipeline reproduces without apt/brew (SPEC §3.4 allows either;
this environment has no lhasa and no apt egress).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from pathlib import Path
from typing import Iterable, Iterator, Protocol

from kyotei.paths import (
    EXTRACTED_DIR,
    LEDGER_PATH,
    RAW_DIR,
    extracted_path,
    fan_extracted_path,
    fan_raw_path,
    raw_path,
)

DAILY_BASE = "https://www1.mbrace.or.jp/od2"
FAN_BASE = "https://www.boatrace.jp/static_extra/pc_static/download/data/kibetsu"

USER_AGENT = "kyotei-ml/0.1 (research; contact via repository)"
SLEEP_SECONDS = 1.0
SLEEP_JITTER = 0.3
REQUEST_TIMEOUT = 60
#: Transient statuses worth one retry with backoff.
RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})


class Status(str, Enum):
    DOWNLOADED = "downloaded"
    SKIPPED = "skipped"  # already on disk
    MISSING = "missing"  # 404: no racing that day (normal)
    KNOWN_MISSING = "known_missing"  # 404 recorded on a previous run
    ERROR = "error"  # anything else -- surfaced, never swallowed


class DownloadError(RuntimeError):
    """Raised for non-404 HTTP failures when the caller asks to fail hard."""


class ExtractError(RuntimeError):
    """Raised when an archive cannot be decompressed."""


# --------------------------------------------------------------------------
# URL construction
# --------------------------------------------------------------------------


def daily_url(kind: str, day: date) -> str:
    """Return the upstream URL for a B or K daily archive.

    >>> daily_url("B", date(2020, 7, 14))
    'https://www1.mbrace.or.jp/od2/B/202007/b200714.lzh'
    """
    if kind not in ("B", "K"):
        raise ValueError(f"kind must be 'B' or 'K', got {kind!r}")
    return f"{DAILY_BASE}/{kind}/{day:%Y%m}/{kind.lower()}{day:%y%m%d}.lzh"


def fan_stem(year: int, half: int) -> str:
    """Half-yearly racer file stem.

    SPEC §3.3: N年前期 = fan{N-1}10, N年後期 = fan{N}04.

    >>> fan_stem(2015, 1), fan_stem(2015, 2)
    ('fan1410', 'fan1504')
    """
    if half == 1:
        return f"fan{(year - 1) % 100:02d}10"
    if half == 2:
        return f"fan{year % 100:02d}04"
    raise ValueError(f"half must be 1 or 2, got {half!r}")


def fan_url(year: int, half: int) -> str:
    """Full URL of a half-yearly racer archive.

    >>> fan_url(2015, 1).endswith("/kibetsu/fan1410.lzh")
    True
    """
    return f"{FAN_BASE}/{fan_stem(year, half)}.lzh"


def daterange(start: date, end: date) -> Iterator[date]:
    """Inclusive date iterator."""
    if end < start:
        raise ValueError(f"end {end} is before start {start}")
    day = start
    while day <= end:
        yield day
        day += timedelta(days=1)


# --------------------------------------------------------------------------
# Ledger -- remembers 404s so re-runs stay cheap and idempotent
# --------------------------------------------------------------------------


class Ledger:
    """Persistent record of request outcomes keyed by ``"<kind>/<iso-date>"``."""

    def __init__(self, path: Path | None = None) -> None:
        # Resolved at call time (not as a default argument) so tests can
        # redirect LEDGER_PATH without writing into the real data directory.
        path = path if path is not None else LEDGER_PATH
        self.path = path
        self._entries: dict[str, str] = {}
        if path.exists():
            self._entries = json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def key(kind: str, day: date) -> str:
        return f"{kind}/{day.isoformat()}"

    def is_missing(self, kind: str, day: date) -> bool:
        return self._entries.get(self.key(kind, day)) == Status.MISSING.value

    def record(self, kind: str, day: date, status: Status) -> None:
        # Only 404s are worth remembering: everything else is decided by
        # whether the file exists on disk.
        if status is Status.MISSING:
            self._entries[self.key(kind, day)] = status.value

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(self._entries, ensure_ascii=False, indent=0, sort_keys=True),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    def __len__(self) -> int:
        return len(self._entries)


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------


class Response(Protocol):
    status_code: int
    content: bytes


class Session(Protocol):
    def get(self, url: str, timeout: int = ...) -> Response: ...


def make_session() -> Session:
    """Real HTTP session. Honours HTTPS_PROXY / REQUESTS_CA_BUNDLE from env."""
    import requests

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session  # type: ignore[return-value]


def _sleep() -> None:
    time.sleep(SLEEP_SECONDS + random.uniform(0, SLEEP_JITTER))


def fetch(
    session: Session,
    url: str,
    dest: Path,
    *,
    sleeper=_sleep,
    retries: int = 1,
) -> tuple[Status, int | None]:
    """Fetch ``url`` into ``dest``. Returns (status, http_status).

    Writes atomically so an interrupted run never leaves a truncated archive
    that a later run would mistake for a complete download.
    """
    attempt = 0
    while True:
        sleeper()
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        code = response.status_code
        if code == 200:
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(response.content)
            tmp.replace(dest)
            return Status.DOWNLOADED, code
        if code == 404:
            return Status.MISSING, code
        if code in RETRYABLE_STATUSES and attempt < retries:
            attempt += 1
            sleeper()
            continue
        return Status.ERROR, code


@dataclass
class Summary:
    """Per-run tally. This is what Phase 1 reports as 取得成功率."""

    counts: Counter = field(default_factory=Counter)
    errors: list[tuple[str, int | None]] = field(default_factory=list)

    def add(self, status: Status, url: str = "", code: int | None = None) -> None:
        self.counts[status.value] += 1
        if status is Status.ERROR:
            self.errors.append((url, code))

    @property
    def attempted(self) -> int:
        return sum(self.counts.values())

    @property
    def available(self) -> int:
        """Files that exist upstream (downloaded now or already held)."""
        return self.counts[Status.DOWNLOADED.value] + self.counts[Status.SKIPPED.value]

    @property
    def success_rate(self) -> float:
        """Share of non-404 targets we actually hold. 404s are excluded because
        a non-race day is not a failure."""
        denom = self.available + self.counts[Status.ERROR.value]
        return self.available / denom if denom else 1.0

    def render(self) -> str:
        lines = [
            f"attempted        : {self.attempted}",
            f"downloaded       : {self.counts[Status.DOWNLOADED.value]}",
            f"skipped (on disk): {self.counts[Status.SKIPPED.value]}",
            f"missing (404 new): {self.counts[Status.MISSING.value]}",
            f"missing (ledger) : {self.counts[Status.KNOWN_MISSING.value]}",
            f"errors           : {self.counts[Status.ERROR.value]}",
            f"success rate     : {self.success_rate:.4%} (excludes 404 non-race days)",
        ]
        if self.errors:
            lines.append("error detail (first 20):")
            lines += [f"  {code} {url}" for url, code in self.errors[:20]]
        return "\n".join(lines)


def download_daily(
    session: Session,
    kind: str,
    start: date,
    end: date,
    *,
    ledger: Ledger | None = None,
    sleeper=_sleep,
    summary: Summary | None = None,
    progress_every: int = 200,
) -> Summary:
    """Download every B or K archive in [start, end], idempotently."""
    ledger = ledger if ledger is not None else Ledger()
    summary = summary if summary is not None else Summary()

    for i, day in enumerate(daterange(start, end), start=1):
        dest = raw_path(kind, day)
        if dest.exists() and dest.stat().st_size > 0:
            summary.add(Status.SKIPPED)
        elif ledger.is_missing(kind, day):
            summary.add(Status.KNOWN_MISSING)
        else:
            url = daily_url(kind, day)
            status, code = fetch(session, url, dest, sleeper=sleeper)
            ledger.record(kind, day, status)
            summary.add(status, url, code)
        if progress_every and i % progress_every == 0:
            print(f"  [{kind}] {day} ... {summary.attempted} processed", flush=True)

    ledger.save()
    return summary


def download_fan(
    session: Session,
    years: Iterable[int],
    *,
    sleeper=_sleep,
    summary: Summary | None = None,
) -> Summary:
    """Download the half-yearly racer files for the given years."""
    summary = summary if summary is not None else Summary()
    for year in years:
        for half in (1, 2):
            stem = fan_stem(year, half)
            dest = fan_raw_path(stem)
            if dest.exists() and dest.stat().st_size > 0:
                summary.add(Status.SKIPPED)
                continue
            url = fan_url(year, half)
            status, code = fetch(session, url, dest, sleeper=sleeper)
            summary.add(status, url, code)
    return summary


# --------------------------------------------------------------------------
# Decompression
# --------------------------------------------------------------------------


def extract_bytes(lzh_path: Path) -> bytes:
    """Return the single fixed-width member of an LZH archive as raw bytes.

    Bytes, not str: the fixed-width layout counts CP932 bytes, and 全角 racer
    names are byte-padded. Decoding before slicing would shift every column.
    """
    import lhafile

    try:
        archive = lhafile.LhaFile(str(lzh_path))
        names = archive.namelist()
    except Exception as exc:  # noqa: BLE001 - re-raised with context, not swallowed
        raise ExtractError(f"cannot open {lzh_path}: {exc!r}") from exc

    if not names:
        raise ExtractError(f"{lzh_path} contains no members")

    payload = b"".join(_read_member(archive, lzh_path, name) for name in sorted(names))
    if not payload:
        raise ExtractError(f"{lzh_path} decompressed to zero bytes")
    return payload


def _read_member(archive, lzh_path: Path, name: str) -> bytes:
    try:
        data = archive.read(name)
    except Exception as exc:  # noqa: BLE001 - context preserved
        raise ExtractError(f"cannot read {name} from {lzh_path}: {exc!r}") from exc
    if data is None:
        raise ExtractError(f"member {name} in {lzh_path} read as None")
    return data


def extract_daily(kind: str, day: date, *, force: bool = False) -> Path | None:
    """Decompress one daily archive to data/extracted/. Returns the path, or
    None when the archive is absent (non-race day)."""
    src = raw_path(kind, day)
    if not src.exists():
        return None
    dest = extracted_path(kind, day)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return dest
    payload = extract_bytes(src)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".txt.part")
    tmp.write_bytes(payload)
    tmp.replace(dest)
    return dest


def extract_all_daily(
    kind: str, start: date, end: date, *, force: bool = False
) -> tuple[int, int, list[tuple[date, str]]]:
    """Decompress a date range. Returns (extracted, absent, failures)."""
    extracted = absent = 0
    failures: list[tuple[date, str]] = []
    for day in daterange(start, end):
        try:
            result = extract_daily(kind, day, force=force)
        except ExtractError as exc:
            failures.append((day, str(exc)))
            continue
        if result is None:
            absent += 1
        else:
            extracted += 1
    return extracted, absent, failures


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parse_day(value: str) -> date:
    return date.fromisoformat(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download boatrace B/K/fan feeds")
    parser.add_argument("--start", type=_parse_day, default=date(2015, 1, 1))
    parser.add_argument("--end", type=_parse_day, default=date.today())
    parser.add_argument(
        "--kind",
        choices=["B", "K", "both"],
        default="both",
        help="which daily feed to fetch",
    )
    parser.add_argument("--fan", action="store_true", help="also fetch fan{YYMM} files")
    parser.add_argument("--extract", action="store_true", help="decompress after fetch")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the targets and exit without any request",
    )
    args = parser.parse_args(argv)

    kinds = ["B", "K"] if args.kind == "both" else [args.kind]

    if args.dry_run:
        days = list(daterange(args.start, args.end))
        print(f"range      : {args.start} .. {args.end} ({len(days)} days)")
        print(f"kinds      : {kinds}")
        print(f"daily target files: {len(days) * len(kinds)}")
        for kind in kinds:
            print(f"  first {kind}: {daily_url(kind, days[0])}")
            print(f"  last  {kind}: {daily_url(kind, days[-1])}")
        if args.fan:
            years = range(args.start.year, args.end.year + 1)
            stems = [fan_stem(y, h) for y in years for h in (1, 2)]
            print(f"fan files  : {len(stems)} -> {stems[0]} .. {stems[-1]}")
        est = len(days) * len(kinds) * (SLEEP_SECONDS + SLEEP_JITTER / 2)
        print(f"est. wall time at {SLEEP_SECONDS}s/request: {est / 3600:.2f} h")
        return 0

    session = make_session()
    ledger = Ledger()
    print(f"ledger: {len(ledger)} known-404 days recorded")

    overall = 0
    for kind in kinds:
        print(f"\n=== {kind} {args.start} .. {args.end} ===", flush=True)
        summary = download_daily(session, kind, args.start, args.end, ledger=ledger)
        print(summary.render())
        overall += summary.counts[Status.ERROR.value]

    if args.fan:
        years = list(range(args.start.year, args.end.year + 1))
        print(f"\n=== fan {years[0]} .. {years[-1]} ===", flush=True)
        fan_summary = download_fan(session, years)
        print(fan_summary.render())
        overall += fan_summary.counts[Status.ERROR.value]

    if args.extract:
        for kind in kinds:
            got, absent, failures = extract_all_daily(kind, args.start, args.end)
            print(f"\n=== extract {kind} ===")
            print(f"extracted: {got}  absent: {absent}  failures: {len(failures)}")
            for day, msg in failures[:20]:
                print(f"  {day}: {msg}")
            overall += len(failures)

    return 1 if overall else 0


if __name__ == "__main__":
    sys.exit(main())
