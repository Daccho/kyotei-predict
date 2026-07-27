"""Leak-free feature construction.

The one rule that matters (SPEC §2.1): every historical aggregate must be
computed from races that started *before* the race being described.

How that is guaranteed structurally
-----------------------------------
Rows are sorted by actual start time, then every "prior" statistic is built as

    prior_sum   = cum_sum().over(group) - current_value
    prior_count = cum_count().over(group) - 1

so the current row is arithmetically removed from its own aggregate, and no
later row has entered the cumulative sum yet. There is no window that could
accidentally include the future, and no join that could pull in a whole-period
total. tests/test_features.py proves it by editing a future row and asserting
every earlier feature is byte-identical.

Rates are shrunk toward an *expanding* global mean rather than the full-sample
mean, because the full-sample mean is itself computed from the future
(SPEC §2 B requires shrinkage for low-denominator motor rates).

A note on 進入コース
-------------------
SPEC §2 A calls the realised course the dominant feature, and it is -- for
explaining a result. It is not known when bets close, so a model that consumes
it cannot be deployed. Both are built here and kept apart:
  * ``PRE_RACE_FEATURES`` -- everything knowable before the deadline, including
    each racer's historical tendency to take a course inside their lane.
  * ``REALISED_FEATURES`` -- adds the realised course. Useful as a diagnostic
    ceiling only; ``model.py`` refuses to use it for backtesting by default.
"""

from __future__ import annotations

import argparse
import io
import sys
from datetime import date, time

import polars as pl

BASE_QUERY = """
SELECT e.race_date, e.stadium_id, e.race_no, e.lane, e.course, e.racer_id,
       e.grade, e.age, e.weight_kg,
       e.national_win_rate, e.national_top2_rate,
       e.local_win_rate, e.local_top2_rate,
       e.motor_no, e.motor_top2_rate, e.boat_no, e.boat_top2_rate,
       e.exhibition_time, e.finish_position, e.finish_status, e.start_timing,
       r.deadline_time, r.fixed_course, r.weather, r.wind_direction,
       r.wind_speed_m, r.wave_height_cm, r.decision, r.series_day, r.distance_m
FROM entries e
JOIN races r USING (race_date, stadium_id, race_no)
WHERE r.has_b AND r.has_k
"""

#: Races are ~30 minutes apart, so ordering by the betting deadline puts a
#: racer's earlier race strictly before their later one on the same day.
ORDER_KEYS = ["race_ts", "stadium_id", "race_no", "lane"]

#: Shrinkage denominators. Larger = trust the population mean for longer.
MOTOR_SHRINKAGE = 50.0
RACER_SHRINKAGE = 30.0
COURSE_SHRINKAGE = 20.0

GRADES = ("A1", "A2", "B1", "B2")
#: Wind directions the K feed publishes. Fixed order so the encoding of a given
#: bearing is identical in train, valid and test -- a rank-based encoding would
#: shift whenever the set of observed values changed.
WIND_DIRECTIONS = (
    "北", "北東", "東", "南東", "南", "南西", "西", "北西",
    "北北東", "東北東", "東南東", "南南東", "南南西", "西南西", "西北西", "北北西",
)
WEATHERS = ("晴", "曇", "曇り", "雨", "雪", "霧")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_frame(dsn: str, query: str = BASE_QUERY) -> pl.DataFrame:
    """Stream a query out of Postgres via COPY and parse it with polars."""
    import psycopg

    buffer = io.BytesIO()
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        with cur.copy(f"COPY ({query}) TO STDOUT WITH (FORMAT csv, HEADER true)") as copy:
            for chunk in copy:
                buffer.write(chunk)
    buffer.seek(0)
    return pl.read_csv(buffer, try_parse_dates=True, infer_schema_length=10_000)


# ---------------------------------------------------------------------------
# Leak-free primitives
# ---------------------------------------------------------------------------


def _prior(value: pl.Expr, group: list[str] | None) -> tuple[pl.Expr, pl.Expr]:
    """Return (prior_sum, prior_count) for ``value``, excluding the current row.

    Nulls contribute to neither, so a blank ST does not count as a 0.0 start.
    """
    ok = value.is_not_null().cast(pl.Int64)
    filled = value.fill_null(0.0)
    if group:
        return (
            filled.cum_sum().over(group) - filled,
            ok.cum_sum().over(group) - ok,
        )
    return filled.cum_sum() - filled, ok.cum_sum() - ok


def _shrunk_rate(
    value: pl.Expr,
    group: list[str],
    shrinkage: float,
    name: str,
) -> list[pl.Expr]:
    """Prior mean of ``value`` within ``group``, shrunk toward the expanding
    global mean of the same quantity.

    Both the group statistic and the global prior exclude the current row, so a
    racer's very first race gets the population mean rather than their own
    outcome.
    """
    group_sum, group_n = _prior(value, group)
    global_sum, global_n = _prior(value, None)
    # Guard the very first rows, where the prior count is still 0.
    global_mean = pl.when(global_n > 0).then(global_sum / global_n).otherwise(None)

    return [
        group_n.alias(f"{name}_n"),
        (
            (group_sum + shrinkage * global_mean)
            / (group_n.cast(pl.Float64) + shrinkage)
        ).alias(f"{name}_rate"),
        # Raw prior mean, kept for diagnostics only: with a handful of starts it
        # is exactly the unstable quantity SPEC §2 B forbids feeding the model.
        pl.when(group_n > 0).then(group_sum / group_n).otherwise(None).alias(f"{name}_raw"),
    ]


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------


def prepare(df: pl.DataFrame) -> pl.DataFrame:
    """Attach the ordering timestamp and the prediction targets."""
    if df.is_empty():
        return df

    deadline = pl.col("deadline_time")
    if df.schema["deadline_time"] == pl.Utf8:
        deadline = deadline.str.strptime(pl.Time, "%H:%M:%S", strict=False)

    return (
        df.with_columns(
            # A missing deadline must not sort before every real race, so it
            # falls back to midday rather than to 00:00.
            deadline.fill_null(time(12, 0)).alias("_deadline"),
        )
        .with_columns(
            pl.col("race_date").dt.combine(pl.col("_deadline")).alias("race_ts"),
            # A null finish means the boat was disqualified, flipped or failed
            # to finish. It definitively did not win, so the label is 0, not
            # null: leaving it null would make it a NaN label in training and
            # would also drop the start from the racer's own denominator, which
            # would quietly flatter anyone who gets disqualified often.
            (pl.col("finish_position") == 1).fill_null(False).cast(pl.Int64).alias("won"),
            (pl.col("finish_position") <= 2).fill_null(False).cast(pl.Int64).alias("top2"),
            (pl.col("finish_position") <= 3).fill_null(False).cast(pl.Int64).alias("top3"),
            # Known before the deadline: how far inside their lane the racer
            # ended up. Used only as history, never for the current race.
            (pl.col("lane") - pl.col("course")).alias("_course_gain"),
        )
        .drop("_deadline")
        .sort(ORDER_KEYS)
    )


def add_history(df: pl.DataFrame) -> pl.DataFrame:
    """Add every backward-looking aggregate. Input must already be sorted."""
    if df.is_empty():
        return df

    won = pl.col("won").cast(pl.Float64)
    top2 = pl.col("top2").cast(pl.Float64)
    gain = pl.col("_course_gain").cast(pl.Float64)
    inside = (pl.col("course") < pl.col("lane")).cast(pl.Float64)

    df = df.with_columns(
        [
            # --- racer form (SPEC §2 B) ---
            *_shrunk_rate(won, ["racer_id"], RACER_SHRINKAGE, "racer_win"),
            *_shrunk_rate(top2, ["racer_id"], RACER_SHRINKAGE, "racer_top2"),
            *_shrunk_rate(
                pl.col("start_timing").cast(pl.Float64), ["racer_id"], RACER_SHRINKAGE, "racer_st"
            ),
            # --- motor, the classic low-denominator trap (SPEC §2 B).
            #     Named _hist_ so it cannot shadow the feed's published
            #     motor_top2_rate column, which is a different quantity. ---
            *_shrunk_rate(
                top2, ["stadium_id", "motor_no"], MOTOR_SHRINKAGE, "motor_hist"
            ),
            # --- lane is known before the deadline; course is not ---
            *_shrunk_rate(won, ["racer_id", "lane"], COURSE_SHRINKAGE, "racer_lane_win"),
            *_shrunk_rate(won, ["stadium_id", "lane"], COURSE_SHRINKAGE, "stadium_lane_win"),
            # --- course-taking tendency: lets the model reason about 進入
            #     without consuming the realised course ---
            *_shrunk_rate(gain, ["racer_id"], COURSE_SHRINKAGE, "racer_course_gain"),
            *_shrunk_rate(inside, ["racer_id"], COURSE_SHRINKAGE, "racer_goes_inside"),
            # --- experience counters ---
            *_shrunk_rate(
                top2, ["racer_id", "stadium_id"], COURSE_SHRINKAGE, "racer_stadium_top2"
            ),
        ]
    )

    # Realised-course history is only meaningful once the course is known, so it
    # is grouped on the current course and therefore belongs to the diagnostic
    # feature set, never to the deployable one.
    df = df.with_columns(
        _shrunk_rate(won, ["racer_id", "course"], COURSE_SHRINKAGE, "racer_course_win")
        + _shrunk_rate(won, ["stadium_id", "course"], COURSE_SHRINKAGE, "stadium_course_win")
    )
    return df


def add_grade_encoding(df: pl.DataFrame) -> pl.DataFrame:
    """Encode the series grade as a rank, strongest meeting first.

    Note the name: ``grade`` in the entries table is the *racer's* class
    (A1/A2/B1/B2), already encoded as ``grade_code``. This is the *series*
    grade, so it is kept strictly separate as ``series_grade``.

    The B/K feeds do not carry it -- it comes from schedule.py -- so the
    column may be absent. It is always created rather than conditionally added,
    because a feature list that silently loses a column between training and
    inference is the kind of mismatch that only shows up in production.

    Rank rather than one-hot: the grades are ordered by prestige and the thing
    that matters, course-1 win rate, moves monotonically along that order
    (オールレディース 0.506 -> 一般 0.542 -> G1 0.613 -> SG 0.622), so one split
    on a rank captures what several one-hot splits would.
    """
    from kyotei.schedule import GRADE_RANK

    if df.is_empty():
        return df
    if "series_grade" not in df.columns:
        return df.with_columns(pl.lit(None, dtype=pl.Int32).alias("series_grade_rank"))
    return df.with_columns(
        pl.col("series_grade")
        .replace_strict(GRADE_RANK, default=None, return_dtype=pl.Int32)
        .alias("series_grade_rank")
    )


def add_encodings(df: pl.DataFrame) -> pl.DataFrame:
    """Categorical encodings and the interaction terms SPEC §2 C asks for."""
    if df.is_empty():
        return df

    grade_map = {g: i for i, g in enumerate(GRADES)}
    wind_map = {w: i for i, w in enumerate(WIND_DIRECTIONS)}
    weather_map = {w: i for i, w in enumerate(WEATHERS)}

    head_wind = pl.col("wind_direction").is_in(["北", "北北東", "北北西"]).cast(pl.Int64)
    wind = pl.col("wind_speed_m").fill_null(0).cast(pl.Float64)

    return df.with_columns(
        pl.col("grade").replace_strict(grade_map, default=None).alias("grade_code"),
        pl.col("wind_direction").replace_strict(wind_map, default=None).alias("wind_dir_code"),
        pl.col("weather").replace_strict(weather_map, default=None).alias("weather_code"),
        # Single weather terms are weak; the interaction with the starting
        # position is the part that carries signal (SPEC §2 C).
        (wind * pl.col("lane")).alias("wind_x_lane"),
        (wind * pl.col("course")).alias("wind_x_course"),
        (head_wind * wind).alias("head_wind_speed"),
        (head_wind * wind * (pl.col("lane") == 1).cast(pl.Int64)).alias("head_wind_x_lane1"),
        (pl.col("wave_height_cm").fill_null(0).cast(pl.Float64) * pl.col("lane"))
        .alias("wave_x_lane"),
        (pl.col("stadium_id") * 10 + pl.col("lane")).alias("stadium_lane"),
        (pl.col("stadium_id") * 10 + pl.col("course").fill_null(0)).alias("stadium_course"),
        pl.col("fixed_course").cast(pl.Int64).alias("fixed_course_flag"),
    )


#: Everything a bettor can know before the deadline.
PRE_RACE_FEATURES = [
    "lane",
    # Race number is known in advance and carries the race's class: late races
    # in a series are 優勝戦/準優勝戦 with stronger fields than early ones. One
    # model covers 1R-12R at all 24 stadiums rather than one model per race.
    "race_no",
    "grade_code",
    "age",
    "weight_kg",
    "national_win_rate",
    "national_top2_rate",
    "local_win_rate",
    "local_top2_rate",
    "motor_top2_rate",
    "boat_top2_rate",
    "exhibition_time",
    "series_day",
    "distance_m",
    # Series grade (SG / G1 / ... / 一般). Published well in advance, so a
    # morning run has it. See schedule.py for why it matters.
    "series_grade_rank",
    "fixed_course_flag",
    "racer_win_rate",
    "racer_win_n",
    "racer_top2_rate",
    "racer_st_rate",
    "motor_hist_n",
    "motor_hist_rate",
    "racer_lane_win_rate",
    "stadium_lane_win_rate",
    "racer_course_gain_rate",
    "racer_goes_inside_rate",
    "racer_stadium_top2_rate",
    "wind_speed_m",
    "wave_height_cm",
    "wind_dir_code",
    "weather_code",
    "wind_x_lane",
    "head_wind_speed",
    "head_wind_x_lane1",
    "wave_x_lane",
    "stadium_id",
    "stadium_lane",
]

#: Adds the realised course. Diagnostic ceiling only -- not bettable.
REALISED_ONLY_FEATURES = [
    "course",
    "stadium_course",
    "wind_x_course",
    "racer_course_win_rate",
    "stadium_course_win_rate",
]
REALISED_FEATURES = PRE_RACE_FEATURES + REALISED_ONLY_FEATURES

#: Columns that only exist once 直前情報 is published (exhibition run and final
#: conditions). In this dataset they arrive via the K feed, i.e. after the race,
#: so a model that runs in the morning cannot have them. SPEC §6 requires the
#: daily pipeline to handle that timing explicitly rather than quietly training
#: on data it will not have at inference time.
LATE_INFORMATION_FEATURES = frozenset(
    {
        "exhibition_time",
        "wind_speed_m",
        "wave_height_cm",
        "wind_dir_code",
        "weather_code",
        "wind_x_lane",
        "head_wind_speed",
        "head_wind_x_lane1",
        "wave_x_lane",
    }
)

#: Everything available from the B feed alone, the morning of the race.
MORNING_FEATURES = [c for c in PRE_RACE_FEATURES if c not in LATE_INFORMATION_FEATURES]

FEATURE_SETS = {
    "morning": MORNING_FEATURES,
    "prerace": PRE_RACE_FEATURES,
    "realised": REALISED_FEATURES,
}


def build(df: pl.DataFrame) -> pl.DataFrame:
    """prepare -> add_history -> add_encodings. Row count is preserved."""
    return add_grade_encoding(add_encodings(add_history(prepare(df))))


def feature_columns(
    feature_set: str = "morning", *, use_realised_course: bool | None = None
) -> list[str]:
    """Columns for a named feature set.

    ``use_realised_course`` is kept as a shorthand so callers that only care
    about the diagnostic-vs-bettable distinction stay readable.
    """
    if use_realised_course is not None:
        feature_set = "realised" if use_realised_course else feature_set
    try:
        return list(FEATURE_SETS[feature_set])
    except KeyError:
        raise ValueError(
            f"unknown feature set {feature_set!r}; choose from {sorted(FEATURE_SETS)}"
        ) from None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    from kyotei.load import DEFAULT_DSN
    from kyotei.paths import PARQUET_DIR

    parser = argparse.ArgumentParser(description="Build the leak-free feature table")
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--out", default=str(PARQUET_DIR / "features.parquet"))
    parser.add_argument("--start", type=date.fromisoformat, default=None)
    parser.add_argument("--end", type=date.fromisoformat, default=None)
    args = parser.parse_args(argv)

    query = BASE_QUERY
    if args.start:
        query += f" AND e.race_date >= '{args.start.isoformat()}'"
    if args.end:
        query += f" AND e.race_date <= '{args.end.isoformat()}'"

    print("loading from postgres ...", flush=True)
    raw = load_frame(args.dsn, query)
    print(f"entries loaded : {raw.height}")

    built = build(raw)
    print(f"feature rows   : {built.height}")
    if built.height != raw.height:
        print(f"!! row count changed: {raw.height} -> {built.height}")
        return 1

    from pathlib import Path

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    built.write_parquet(out)
    print(f"written        : {out} ({out.stat().st_size / 1e6:.1f} MB)")

    print("\n-- null share of model features --")
    for column in feature_columns():
        if column in built.columns:
            share = built[column].null_count() / built.height
            if share > 0.01:
                print(f"  {column:28s} {share:6.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
