"""Parse raw archives straight to parquet, with no database (Phase 1 output).

Two reasons this exists alongside load.py:

  * Portability. The parsed tables are the artefact worth keeping: the raw LZH
    files are re-downloadable, a Postgres cluster in an ephemeral container is
    not. Parquet travels to Colab, to Drive, or into the repo.
  * The by-year parse report. This is the Phase 1 completion evidence, and it
    should not require a database to produce.

Output (data/parquet/):
    entries.parquet   one row per boat per race, joined to its race metadata,
                      matching the column set features.BASE_QUERY produces
    payouts.parquet   every dividend, the basis of the backtest
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import polars as pl

from kyotei.load import ENTRY_COLUMNS, PAYOUT_COLUMNS, RACE_COLUMNS, parse_day
from kyotei.parse_stats import ParseStats
from kyotei.paths import PARQUET_DIR

#: Race-level columns carried onto every entry row, mirroring the SQL join.
RACE_CONTEXT = [
    "deadline_time", "fixed_course", "weather", "wind_direction", "wind_speed_m",
    "wave_height_cm", "decision", "series_day", "distance_m", "title",
    "has_b", "has_k",
]

RACE_KEYS = ["race_date", "stadium_id", "race_no"]

#: Explicit dtypes rather than inference. Inference reads only the first rows,
#: and columns like finish_status are null for thousands of rows before the
#: first 'F'/'K0'/'S1' appears -- which then fails to append to an inferred
#: numeric builder. Declaring the schema also keeps parquet stable across runs.
_DTYPES: dict[str, pl.DataType] = {
    "race_date": pl.Date,
    "stadium_id": pl.Int16,
    "race_no": pl.Int16,
    "lane": pl.Int8,
    "course": pl.Int8,
    "racer_id": pl.Int32,
    "racer_name": pl.Utf8,
    "grade": pl.Utf8,
    "branch": pl.Utf8,
    "age": pl.Int16,
    "weight_kg": pl.Float64,
    "national_win_rate": pl.Float64,
    "national_top2_rate": pl.Float64,
    "local_win_rate": pl.Float64,
    "local_top2_rate": pl.Float64,
    "motor_no": pl.Int16,
    "motor_top2_rate": pl.Float64,
    "boat_no": pl.Int16,
    "boat_top2_rate": pl.Float64,
    "exhibition_time": pl.Float64,
    "finish_position": pl.Int8,
    "finish_status": pl.Utf8,
    "start_timing": pl.Float64,
    "race_time_sec": pl.Float64,
    # race-level
    "title": pl.Utf8,
    "distance_m": pl.Int16,
    "deadline_time": pl.Time,
    "series_day": pl.Int16,
    "fixed_course": pl.Boolean,
    "weather": pl.Utf8,
    "wind_direction": pl.Utf8,
    "wind_speed_m": pl.Int16,
    "wave_height_cm": pl.Int16,
    "decision": pl.Utf8,
    "has_b": pl.Boolean,
    "has_k": pl.Boolean,
    # payouts
    "bet_type": pl.Utf8,
    "combination": pl.Utf8,
    "payout_yen": pl.Int32,
    "popularity": pl.Int16,
}


def _frame(rows: list[dict], columns) -> pl.DataFrame:
    schema = {c: _DTYPES[c] for c in columns}
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema)


def parse_range(
    start: date, end: date, *, progress_every: int = 200
) -> tuple[pl.DataFrame, pl.DataFrame, ParseStats, ParseStats]:
    """Parse every day in range into flat frames plus per-year parse stats."""
    from kyotei.download import daterange

    b_stats, k_stats = ParseStats("B"), ParseStats("K")
    races: list[dict] = []
    entries: list[dict] = []
    payouts: list[dict] = []

    for index, day in enumerate(daterange(start, end), start=1):
        day_races, day_entries, day_payouts, b_outcome, k_outcome = parse_day(day)
        races += day_races
        entries += day_entries
        payouts += day_payouts
        if b_outcome.records_ok or b_outcome.records_failed or b_outcome.fatal:
            b_stats.add(b_outcome)
        if k_outcome.records_ok or k_outcome.records_failed or k_outcome.fatal:
            k_stats.add(k_outcome)
        if progress_every and index % progress_every == 0:
            print(f"  parsed through {day}: {len(entries)} entries", flush=True)

    return (
        _frame(races, RACE_COLUMNS),
        _frame(entries, ENTRY_COLUMNS),
        _frame(payouts, PAYOUT_COLUMNS),
        b_stats,
        k_stats,
    )


def join_entries(races: pl.DataFrame, entries: pl.DataFrame) -> pl.DataFrame:
    """Attach race metadata to every entry, checking the row count is preserved.

    A join that changes the row count is exactly the silent failure SPEC §7
    forbids, so it is asserted rather than hoped for.
    """
    if races.is_empty() or entries.is_empty():
        return entries

    context = [c for c in RACE_CONTEXT if c in races.columns]
    before = entries.height
    joined = entries.join(races.select([*RACE_KEYS, *context]), on=RACE_KEYS, how="left")
    if joined.height != before:
        raise ValueError(
            f"joining race metadata changed the row count: {before} -> {joined.height}"
        )
    return joined


def race_entry_consistency(joined: pl.DataFrame) -> pl.DataFrame:
    """Per-year counts and the checks Phase 1 has to report."""
    if joined.is_empty():
        return pl.DataFrame()
    year = pl.col("race_date").dt.year().alias("year")
    per_race = joined.group_by(RACE_KEYS).agg(pl.len().alias("boats"))
    bad = per_race.filter(pl.col("boats") != 6)

    summary = (
        joined.with_columns(year)
        .group_by("year")
        .agg(
            pl.struct(RACE_KEYS).n_unique().alias("races"),
            pl.len().alias("entries"),
            pl.col("course").is_not_null().sum().alias("with_course"),
            pl.col("finish_position").is_not_null().sum().alias("with_finish"),
            ((pl.col("course") != pl.col("lane")).sum()).alias("course_differs"),
            (
                (pl.col("finish_position") == 1) & (pl.col("course") == 1)
            ).sum().alias("course1_wins"),
            (pl.col("finish_position") == 1).sum().alias("winners"),
        )
        .sort("year")
        .with_columns(
            (pl.col("entries") / pl.col("races")).alias("boats_per_race"),
            (pl.col("course1_wins") / pl.col("winners")).alias("course1_win_rate"),
            (pl.col("course_differs") / pl.col("with_course")).alias("course_differs_share"),
        )
    )
    print(f"\nraces whose entry count != 6 : {bad.height}")
    if bad.height:
        print(bad.head(10))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parse raw archives to parquet")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2015, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date.today())
    parser.add_argument("--out", default=str(PARQUET_DIR))
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"=== parsing {args.start} .. {args.end} ===", flush=True)
    races, entries, payouts, b_stats, k_stats = parse_range(args.start, args.end)
    print(f"\nraces   : {races.height}")
    print(f"entries : {entries.height}")
    print(f"payouts : {payouts.height}")

    print("\n=== B parse success by year ===")
    print(b_stats.render())
    print("\n=== K parse success by year ===")
    print(k_stats.render())

    if entries.is_empty():
        print("\n!! nothing parsed; download the archives first")
        return 1

    joined = join_entries(races, entries)
    print(f"\njoined  : {joined.height} rows (entries were {entries.height})")

    summary = race_entry_consistency(joined)
    if not summary.is_empty():
        print("\n=== per-year summary ===")
        with pl.Config(tbl_rows=30, float_precision=4, tbl_width_chars=220):
            print(summary)

    joined.write_parquet(out / "entries.parquet")
    payouts.write_parquet(out / "payouts.parquet")
    races.write_parquet(out / "races.parquet")
    for name in ("entries", "payouts", "races"):
        path = out / f"{name}.parquet"
        print(f"written : {path} ({path.stat().st_size / 1e6:.1f} MB)")

    suspects = b_stats.layout_change_suspects() + k_stats.layout_change_suspects()
    if suspects:
        print(f"\n!! years below the parse-rate threshold: {sorted(set(suspects))}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
