"""Backfill real 締切時オッズ for a season, resumably.

Why this is shaped the way it is
--------------------------------
One request per race, ~53,000 races in a year, 1 req/s to the site = ~15 hours.
That is longer than an ephemeral container lives, so two properties matter more
than speed:

**Any prefix must be usable.** Races are visited in a fixed-seed *shuffled*
order, not chronologically. If the job is interrupted after 30%, a chronological
sweep would leave January-April only -- seasonally biased, and worthless for an
honest ROI estimate. A deterministic shuffle makes whatever fraction completed
an unbiased random sample of the whole year, which is directly usable. The seed
is fixed so a resumed run continues the same sequence rather than re-drawing it.

**It must resume.** Results append to JSONL as they arrive, and the set of
already-fetched races is read back on startup. Races whose odds are not retained
are recorded too, so they are not retried on every run.

Politeness. The archive downloader budgets 1 req/s against www1.mbrace.or.jp;
this budgets 1 req/s against www.boatrace.jp. They are different hosts, so the
per-host load is unchanged when both run.

Raw HTML is not kept: 53,000 pages x ~45KB is 2.4GB for data we parse once. Only
the 120 parsed prices per race are stored.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
from concurrent import futures
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl

from kyotei import parse_b
from kyotei import scrape as sc
from kyotei.download import ExtractError, RateLimiter, SessionPool, daterange, extract_bytes
from kyotei.paths import DATA_DIR, raw_path

ODDS_DIR = DATA_DIR / "odds"

#: Fixed so an interrupted run resumes the same visiting order, and so any
#: prefix is reproducibly the same unbiased sample.
SHUFFLE_SEED = 20240101

DEFAULT_WORKERS = 12


@dataclass(frozen=True)
class RaceRef:
    day: date
    stadium_id: int
    race_no: int

    def key(self) -> str:
        return f"{self.day.isoformat()}/{self.stadium_id:02d}/{self.race_no}"


# ---------------------------------------------------------------------------
# Which races exist
# ---------------------------------------------------------------------------


def race_list_from_archive(start: date, end: date) -> list[RaceRef]:
    """Read the race card archive to learn which races actually ran.

    Brute-forcing 24 stadiums x 12 races would double the request count and
    still miss nothing the B feed does not already state.
    """
    races: list[RaceRef] = []
    for day in daterange(start, end):
        path = raw_path("B", day)
        if not path.exists():
            continue
        try:
            payload = extract_bytes(path)
        except ExtractError as exc:
            print(f"  {day}: unreadable archive ({exc})")
            continue
        parsed, _ = parse_b.parse_payload(payload, day, f"b{day:%y%m%d}")
        races.extend(RaceRef(day, r.stadium_id, r.race_no) for r in parsed)
    return races


def visiting_order(races: list[RaceRef], seed: int = SHUFFLE_SEED) -> list[RaceRef]:
    """Deterministic shuffle, so any prefix is an unbiased sample of the period."""
    ordered = sorted(races, key=lambda r: (r.day, r.stadium_id, r.race_no))
    random.Random(seed).shuffle(ordered)
    return ordered


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def jsonl_path(year: int, directory: Path | None = None) -> Path:
    return (directory or ODDS_DIR) / f"odds3t_{year}.jsonl"


def load_done(path: Path) -> set[str]:
    """Keys already recorded, including ones known to be unavailable."""
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A partially written final line from a killed run: ignore it
                # rather than crashing, and let the race be refetched.
                continue
            if "k" in record:
                done.add(record["k"])
    return done


class Writer:
    """Append-only JSONL writer, flushed per record so a kill loses nothing."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self._handle.write(line + "\n")
            self._handle.flush()

    def close(self) -> None:
        self._handle.close()


# ---------------------------------------------------------------------------
# Backfill
# ---------------------------------------------------------------------------


@dataclass
class Progress:
    fetched: int = 0
    unavailable: int = 0
    errors: int = 0

    @property
    def attempted(self) -> int:
        return self.fetched + self.unavailable + self.errors


def backfill(
    races: list[RaceRef],
    out: Path,
    *,
    session=None,
    limiter: RateLimiter | None = None,
    workers: int = DEFAULT_WORKERS,
    limit: int | None = None,
    progress_every: int = 200,
) -> Progress:
    """Fetch odds for ``races`` (already in visiting order), skipping done ones."""
    done = load_done(out)
    todo = [r for r in races if r.key() not in done]
    if limit is not None:
        todo = todo[:limit]
    print(f"races known: {len(races)}   already recorded: {len(done)}   to fetch: {len(todo)}")
    if not todo:
        return Progress()

    session = session if session is not None else SessionPool()
    limiter = limiter if limiter is not None else RateLimiter(1.0)
    writer = Writer(out)
    progress = Progress()
    lock = threading.Lock()

    def work(ref: RaceRef) -> dict:
        try:
            odds = sc.fetch_trifecta_odds(
                session, ref.day, ref.stadium_id, ref.race_no, sleeper=limiter.acquire
            )
        except sc.PageUnavailable as exc:
            return {"k": ref.key(), "d": ref.day.isoformat(), "j": ref.stadium_id,
                    "r": ref.race_no, "unavailable": str(exc)[:120]}
        except Exception as exc:  # noqa: BLE001 - recorded, not silently dropped
            return {"k": ref.key(), "d": ref.day.isoformat(), "j": ref.stadium_id,
                    "r": ref.race_no, "error": f"{type(exc).__name__}: {exc}"[:160]}
        return {"k": ref.key(), "d": ref.day.isoformat(), "j": ref.stadium_id,
                "r": ref.race_no, "o": odds}

    try:
        with futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for record in pool.map(work, todo):
                writer.write(record)
                with lock:
                    if "o" in record:
                        progress.fetched += 1
                    elif "unavailable" in record:
                        progress.unavailable += 1
                    else:
                        progress.errors += 1
                    if progress_every and progress.attempted % progress_every == 0:
                        print(
                            f"  {progress.attempted}/{len(todo)}  "
                            f"ok={progress.fetched} unavailable={progress.unavailable} "
                            f"errors={progress.errors}",
                            flush=True,
                        )
    finally:
        writer.close()
    return progress


# ---------------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------------


def to_frame(path: Path) -> pl.DataFrame:
    """Long-format frame: one row per (race, combination)."""
    rows: list[dict] = []
    if not path.exists():
        return pl.DataFrame(
            schema={
                "race_date": pl.Date, "stadium_id": pl.Int16, "race_no": pl.Int16,
                "combination": pl.Utf8, "odds": pl.Float64,
            }
        )
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "o" not in record:
                continue
            day = date.fromisoformat(record["d"])
            for combination, price in record["o"].items():
                rows.append(
                    {
                        "race_date": day,
                        "stadium_id": int(record["j"]),
                        "race_no": int(record["r"]),
                        "combination": combination,
                        "odds": float(price),
                    }
                )
    frame = pl.DataFrame(rows)
    if frame.is_empty():
        return frame
    return frame.with_columns(
        pl.col("stadium_id").cast(pl.Int16), pl.col("race_no").cast(pl.Int16)
    )


def coverage(path: Path) -> pl.DataFrame:
    """How much of each month has been fetched -- the check that a partial run is
    still spread across the season rather than bunched at one end."""
    frame = to_frame(path)
    if frame.is_empty():
        return frame
    return (
        frame.select(["race_date", "stadium_id", "race_no"])
        .unique()
        .with_columns(pl.col("race_date").dt.month().alias("month"))
        .group_by("month")
        .agg(pl.len().alias("races"))
        .sort("month")
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill real trifecta odds")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--out-dir", default=str(ODDS_DIR))
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--rate", type=float, default=1.0,
                        help="requests per second against www.boatrace.jp")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after this many new races (a prefix is still "
                             "an unbiased sample of the year)")
    parser.add_argument("--compact", action="store_true",
                        help="write the parquet and exit without fetching")
    args = parser.parse_args(argv)

    out = jsonl_path(args.year, Path(args.out_dir))
    parquet = out.with_suffix(".parquet")

    if not args.compact:
        start, end = date(args.year, 1, 1), date(args.year, 12, 31)
        print(f"=== reading the {args.year} race card archive ===", flush=True)
        races = race_list_from_archive(start, end)
        if not races:
            print(f"!! no B archives for {args.year}; download them first:")
            print(f"   python -m kyotei.download --start {start} --end {end} --kind B")
            return 1
        ordered = visiting_order(races)
        print(f"races: {len(ordered)} (visited in fixed-seed shuffled order, so an "
              f"interrupted run is still a random sample)")

        progress = backfill(
            ordered, out,
            limiter=RateLimiter(args.rate),
            workers=args.workers,
            limit=args.limit,
        )
        print(f"\nfetched={progress.fetched} unavailable={progress.unavailable} "
              f"errors={progress.errors}")

    frame = to_frame(out)
    print(f"\nrecords: {frame.height} rows "
          f"({frame.height / 120:.0f} races x 120 combinations)")
    if not frame.is_empty():
        frame.write_parquet(parquet)
        print(f"written: {parquet} ({parquet.stat().st_size / 1e6:.1f} MB)")
        print("\nmonthly coverage (a partial run should be spread, not bunched):")
        print(coverage(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
