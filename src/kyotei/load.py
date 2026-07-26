"""Parse the raw archives and load them into Postgres.

B supplies the roster (who is in which lane, with their published form) and K
supplies what happened (actual course, finish, ST, dividends). Both describe the
same race, so rows are merged per race key in Python and then staged with COPY
and upserted. That keeps the load idempotent: re-running a day updates instead
of duplicating, which matters because K for the most recent day may arrive after
B has already been loaded.

Everything is reported, nothing is silently dropped:
  * per-year parse success rates (SPEC §3.2 layout-change detection)
  * row counts at each stage, so a join that loses rows is visible
  * validation queries run against the schema's views
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from typing import Any, Iterable, Iterator

from kyotei.download import ExtractError, daterange, extract_bytes
from kyotei.parse_stats import FileOutcome, ParseStats
from kyotei.paths import raw_path
from kyotei import parse_b, parse_k

DEFAULT_DSN = os.environ.get("KYOTEI_DSN", "postgresql://kyotei:kyotei@localhost:5432/kyotei")

RACE_COLUMNS = (
    "race_date", "stadium_id", "race_no", "title", "distance_m", "deadline_time",
    "series_day", "fixed_course", "weather", "wind_direction", "wind_speed_m",
    "wave_height_cm", "decision", "has_b", "has_k",
)
ENTRY_COLUMNS = (
    "race_date", "stadium_id", "race_no", "lane", "course", "racer_id", "racer_name",
    "grade", "branch", "age", "weight_kg", "national_win_rate", "national_top2_rate",
    "local_win_rate", "local_top2_rate", "motor_no", "motor_top2_rate", "boat_no",
    "boat_top2_rate", "exhibition_time", "finish_position", "finish_status",
    "start_timing", "race_time_sec",
)
PAYOUT_COLUMNS = (
    "race_date", "stadium_id", "race_no", "bet_type", "combination",
    "payout_yen", "popularity",
)


# ---------------------------------------------------------------------------
# Parse + merge
# ---------------------------------------------------------------------------


def parse_day(day: date) -> tuple[list[dict], list[dict], list[dict], FileOutcome, FileOutcome]:
    """Parse one day's B and K archives and merge them into flat row lists."""
    races: dict[tuple, dict[str, Any]] = {}
    entries: dict[tuple, dict[str, Any]] = {}
    payouts: list[dict[str, Any]] = []

    b_outcome = _read(day, "B", races, entries, payouts)
    k_outcome = _read(day, "K", races, entries, payouts)

    return list(races.values()), list(entries.values()), payouts, b_outcome, k_outcome


def _read(
    day: date,
    kind: str,
    races: dict[tuple, dict],
    entries: dict[tuple, dict],
    payouts: list[dict],
) -> FileOutcome:
    source = f"{kind.lower()}{day:%y%m%d}"
    path = raw_path(kind, day)
    if not path.exists():
        # No archive: a non-race day. Not an error, and not a parsed file either.
        return FileOutcome(source=source, day=day, fatal=None)

    try:
        payload = extract_bytes(path)
    except ExtractError as exc:
        return FileOutcome(source=source, day=day, fatal=str(exc))

    module = parse_b if kind == "B" else parse_k
    parsed, outcome = module.parse_payload(payload, day, source)

    for race in parsed:
        # NOT NULL booleans must start False, not None: dict.fromkeys would set
        # them to None and setdefault cannot fix an existing None key.
        row = races.setdefault(
            race.key,
            {**dict.fromkeys(RACE_COLUMNS), "has_b": False, "has_k": False,
             "fixed_course": False},
        )
        row["race_date"], row["stadium_id"], row["race_no"] = race.key
        row[f"has_{kind.lower()}"] = True
        for column in RACE_COLUMNS:
            value = getattr(race, column, None)
            if value is not None and row.get(column) is None:
                row[column] = value
        row["fixed_course"] = bool(row.get("fixed_course"))

        for entry in race.entries:
            lane = entry.get("lane")
            if lane is None:
                continue
            key = (*race.key, lane)
            merged = entries.setdefault(key, dict.fromkeys(ENTRY_COLUMNS))
            merged["race_date"], merged["stadium_id"], merged["race_no"] = race.key
            merged["lane"] = lane
            for column, value in entry.items():
                if column in ENTRY_COLUMNS and value is not None:
                    merged[column] = value

        for payout in getattr(race, "payouts", []):
            payouts.append({**payout, "race_date": race.key[0],
                            "stadium_id": race.key[1], "race_no": race.key[2]})

    return outcome


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

STAGING_DDL = """
CREATE TEMP TABLE stg_races (LIKE races) ON COMMIT DROP;
CREATE TEMP TABLE stg_entries (LIKE entries) ON COMMIT DROP;
CREATE TEMP TABLE stg_payouts (LIKE payouts) ON COMMIT DROP;
"""

UPSERT_RACES = f"""
INSERT INTO races ({','.join(RACE_COLUMNS)})
SELECT {','.join(RACE_COLUMNS)} FROM stg_races
ON CONFLICT (race_date, stadium_id, race_no) DO UPDATE SET
  title = COALESCE(EXCLUDED.title, races.title),
  distance_m = COALESCE(EXCLUDED.distance_m, races.distance_m),
  deadline_time = COALESCE(EXCLUDED.deadline_time, races.deadline_time),
  series_day = COALESCE(EXCLUDED.series_day, races.series_day),
  fixed_course = races.fixed_course OR EXCLUDED.fixed_course,
  weather = COALESCE(EXCLUDED.weather, races.weather),
  wind_direction = COALESCE(EXCLUDED.wind_direction, races.wind_direction),
  wind_speed_m = COALESCE(EXCLUDED.wind_speed_m, races.wind_speed_m),
  wave_height_cm = COALESCE(EXCLUDED.wave_height_cm, races.wave_height_cm),
  decision = COALESCE(EXCLUDED.decision, races.decision),
  has_b = races.has_b OR EXCLUDED.has_b,
  has_k = races.has_k OR EXCLUDED.has_k
"""

_ENTRY_UPDATES = ",\n  ".join(
    f"{c} = COALESCE(EXCLUDED.{c}, entries.{c})"
    for c in ENTRY_COLUMNS
    if c not in ("race_date", "stadium_id", "race_no", "lane")
)
UPSERT_ENTRIES = f"""
INSERT INTO entries ({','.join(ENTRY_COLUMNS)})
SELECT {','.join(ENTRY_COLUMNS)} FROM stg_entries
ON CONFLICT (race_date, stadium_id, race_no, lane) DO UPDATE SET
  {_ENTRY_UPDATES}
"""

UPSERT_PAYOUTS = f"""
INSERT INTO payouts ({','.join(PAYOUT_COLUMNS)})
SELECT DISTINCT ON (race_date, stadium_id, race_no, bet_type, combination)
  {','.join(PAYOUT_COLUMNS)} FROM stg_payouts
ON CONFLICT (race_date, stadium_id, race_no, bet_type, combination) DO UPDATE SET
  payout_yen = EXCLUDED.payout_yen,
  popularity = COALESCE(EXCLUDED.popularity, payouts.popularity)
"""


def _copy(cur, table: str, columns: tuple[str, ...], rows: Iterable[dict]) -> int:
    count = 0
    with cur.copy(f"COPY {table} ({','.join(columns)}) FROM STDIN") as copy:
        for row in rows:
            copy.write_row([row.get(c) for c in columns])
            count += 1
    return count


def load_range(
    dsn: str,
    start: date,
    end: date,
    *,
    batch_days: int = 30,
    progress_every: int = 90,
) -> tuple[ParseStats, ParseStats, dict[str, int]]:
    import psycopg

    b_stats, k_stats = ParseStats("B"), ParseStats("K")
    totals = {"races": 0, "entries": 0, "payouts": 0}

    with psycopg.connect(dsn) as conn:
        for chunk in _chunks(daterange(start, end), batch_days):
            races: list[dict] = []
            entries: list[dict] = []
            payouts: list[dict] = []
            for day in chunk:
                r, e, p, bo, ko = parse_day(day)
                races += r
                entries += e
                payouts += p
                if bo.records_ok or bo.records_failed or bo.fatal:
                    b_stats.add(bo)
                if ko.records_ok or ko.records_failed or ko.fatal:
                    k_stats.add(ko)

            if not races:
                continue
            with conn.cursor() as cur:
                cur.execute(STAGING_DDL)
                totals["races"] += _copy(cur, "stg_races", RACE_COLUMNS, races)
                totals["entries"] += _copy(cur, "stg_entries", ENTRY_COLUMNS, entries)
                totals["payouts"] += _copy(cur, "stg_payouts", PAYOUT_COLUMNS, payouts)
                cur.execute(UPSERT_RACES)
                cur.execute(UPSERT_ENTRIES)
                cur.execute(UPSERT_PAYOUTS)
            conn.commit()
            if progress_every:
                print(
                    f"  loaded through {chunk[-1]}  "
                    f"races={totals['races']} entries={totals['entries']}",
                    flush=True,
                )

    return b_stats, k_stats, totals


def _chunks(it: Iterable[date], size: int) -> Iterator[list[date]]:
    batch: list[date] = []
    for item in it:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


# ---------------------------------------------------------------------------
# Validation -- the Phase 1 completion evidence
# ---------------------------------------------------------------------------

VALIDATIONS: tuple[tuple[str, str], ...] = (
    ("races / entries total", "SELECT (SELECT count(*) FROM races), (SELECT count(*) FROM entries)"),
    ("races whose entry count != 6", "SELECT count(*) FROM v_bad_entry_counts"),
    ("entries with NULL course", "SELECT count(*) FROM entries WHERE course IS NULL"),
    ("entries where course <> lane", "SELECT count(*) FROM entries WHERE course <> lane"),
    ("races with results but no trifecta", "SELECT count(*) FROM v_missing_trifecta"),
    ("distinct stadiums", "SELECT count(DISTINCT stadium_id) FROM races"),
    ("payout rows by type",
     "SELECT bet_type, count(*) FROM payouts GROUP BY 1 ORDER BY 2 DESC"),
    ("per-year summary", "SELECT * FROM v_yearly_summary"),
    ("course-1 win rate by year", "SELECT * FROM v_course1_winrate"),
)


def validate(dsn: str) -> int:
    """Run every validation query and print the raw numbers."""
    import psycopg

    failures = 0
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        for label, sql in VALIDATIONS:
            cur.execute(sql)
            rows = cur.fetchall()
            print(f"\n## {label}")
            if cur.description:
                print("   " + " | ".join(d.name for d in cur.description))
            for row in rows:
                print("   " + " | ".join("-" if v is None else str(v) for v in row))
            if label.startswith(("races whose entry count", "races with results but no")):
                if rows and rows[0][0]:
                    failures += 1
                    print(f"   !! expected 0, got {rows[0][0]}")
    return failures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parse raw archives into Postgres")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2015, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date.today())
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    if not args.validate_only:
        print(f"=== loading {args.start} .. {args.end} ===", flush=True)
        b_stats, k_stats, totals = load_range(args.dsn, args.start, args.end)
        print("\n=== B parse success by year ===")
        print(b_stats.render())
        print("\n=== K parse success by year ===")
        print(k_stats.render())
        print(f"\nstaged rows: {totals}")

    print("\n=== validation ===")
    failures = validate(args.dsn)
    print(f"\nhard validation failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
