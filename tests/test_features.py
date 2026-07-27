"""Leak tests for the feature builder.

SPEC §2.1 demands proof, not assertion: "実装後、リークが無いことを検証する
テストを必ず書く（例：未来のレコードを改変してもトレーニングデータが変化しない
こと）". That exact experiment is test_editing_the_future_does_not_change_the_past.

The fixtures are synthetic so the expected numbers can be computed by hand.
"""

from __future__ import annotations

from datetime import date, time, timedelta

import polars as pl
import pytest

from kyotei import features as ft


def make_races(rows: list[dict]) -> pl.DataFrame:
    """Build a minimal entries+races frame with sensible defaults."""
    defaults = {
        "race_date": date(2020, 1, 1),
        "stadium_id": 1,
        "race_no": 1,
        "lane": 1,
        "course": 1,
        "racer_id": 1000,
        "grade": "A1",
        "age": 30,
        "weight_kg": 52.0,
        "national_win_rate": 5.0,
        "national_top2_rate": 30.0,
        "local_win_rate": 5.0,
        "local_top2_rate": 30.0,
        "motor_no": 10,
        "motor_top2_rate": 35.0,
        "boat_no": 20,
        "boat_top2_rate": 35.0,
        "exhibition_time": 6.80,
        "finish_position": 1,
        "finish_status": None,
        "start_timing": 0.15,
        "deadline_time": time(12, 0),
        "fixed_course": False,
        "weather": "晴",
        "wind_direction": "北",
        "wind_speed_m": 2,
        "wave_height_cm": 1,
        "decision": "逃げ",
        "series_day": 1,
        "distance_m": 1800,
    }
    return pl.DataFrame([{**defaults, **row} for row in rows])


BASE_DAY = date(2020, 1, 1)


def one_racer_sequence(finishes: list[int]) -> pl.DataFrame:
    """One racer, one race per day, with the given finishing positions."""
    return make_races(
        [
            {
                "race_date": BASE_DAY + timedelta(days=i),
                "race_no": 1,
                "finish_position": pos,
            }
            for i, pos in enumerate(finishes)
        ]
    )


def six_boat_days(days: int) -> pl.DataFrame:
    """Realistic population: full 6-boat races where lane N finishes Nth.

    Gives a global win rate of 1/6, so shrinkage has somewhere to shrink to.
    Racer 101 is always in lane 1 and always wins.
    """
    return make_races(
        [
            {
                "race_date": BASE_DAY + timedelta(days=d),
                "race_no": 1,
                "racer_id": 100 + lane,
                "motor_no": 100 + lane,
                "lane": lane,
                "course": lane,
                "finish_position": lane,
            }
            for d in range(days)
            for lane in range(1, 7)
        ]
    )


# ---------------------------------------------------------------------------
# The headline leak test
# ---------------------------------------------------------------------------


def test_editing_the_future_does_not_change_the_past():
    """Mutate the last race's outcome; every earlier feature row must be identical."""
    base = one_racer_sequence([1, 2, 1, 3, 1, 4])

    original = ft.build(base)

    tampered_input = base.with_columns(
        pl.when(pl.col("race_date") == date(2020, 1, 6))
        .then(pl.lit(6))
        .otherwise(pl.col("finish_position"))
        .alias("finish_position")
    )
    tampered = ft.build(tampered_input)

    earlier = pl.col("race_date") < date(2020, 1, 6)
    left = original.filter(earlier)
    right = tampered.filter(earlier)

    assert left.height == 5
    for column in ft.PRE_RACE_FEATURES:
        if column in left.columns:
            assert left[column].to_list() == right[column].to_list(), (
                f"feature {column!r} changed when a FUTURE race was edited -- leak"
            )


def test_editing_the_future_does_not_change_realised_course_features_either():
    base = one_racer_sequence([1, 2, 1, 3, 1, 4])
    original = ft.build(base)
    tampered = ft.build(
        base.with_columns(
            pl.when(pl.col("race_date") == date(2020, 1, 6))
            .then(pl.lit(4))
            .otherwise(pl.col("course"))
            .alias("course")
        )
    )
    earlier = pl.col("race_date") < date(2020, 1, 6)
    for column in ft.REALISED_ONLY_FEATURES:
        if column in original.columns:
            assert (
                original.filter(earlier)[column].to_list()
                == tampered.filter(earlier)[column].to_list()
            ), f"{column!r} leaked a future course"


def test_appending_future_races_does_not_change_existing_rows():
    """Re-running with more recent data must not restate history."""
    short = one_racer_sequence([1, 2, 3])
    long = one_racer_sequence([1, 2, 3, 1, 1, 1])

    a = ft.build(short)
    b = ft.build(long).head(3)

    for column in ft.PRE_RACE_FEATURES:
        if column in a.columns:
            assert a[column].to_list() == b[column].to_list(), f"{column!r} restated"


# ---------------------------------------------------------------------------
# The current row must never be inside its own aggregate
# ---------------------------------------------------------------------------


def test_first_race_has_no_own_history():
    out = ft.build(one_racer_sequence([1, 1, 1]))
    assert out["racer_win_n"].to_list() == [0, 1, 2]
    # A first-timer falls back to the population prior, not to their own result.
    assert out["racer_win_raw"][0] is None


def test_prior_counts_exclude_the_current_race():
    out = ft.build(one_racer_sequence([1, 2, 3, 4, 5, 6]))
    assert out["racer_win_n"].to_list() == [0, 1, 2, 3, 4, 5]


def test_prior_win_rate_uses_only_earlier_results():
    """Wins on days 1-3, losses after: the raw prior must lag by one race."""
    out = ft.build(one_racer_sequence([1, 1, 1, 6, 6, 6]))
    raw = out["racer_win_raw"].to_list()

    assert raw[0] is None                  # no history
    assert raw[1] == pytest.approx(1.0)    # 1 win in 1
    assert raw[3] == pytest.approx(1.0)    # 3 wins in 3, day 4 not counted yet
    assert raw[4] == pytest.approx(0.75)   # day 4's loss now included
    assert raw[5] == pytest.approx(0.6)


def test_two_racers_do_not_share_history():
    df = make_races(
        [
            {"race_date": date(2020, 1, 1), "racer_id": 1, "finish_position": 1},
            {"race_date": date(2020, 1, 1), "racer_id": 2, "lane": 2, "finish_position": 2},
            {"race_date": date(2020, 1, 2), "race_no": 2, "racer_id": 1, "finish_position": 1},
            {"race_date": date(2020, 1, 2), "race_no": 2, "racer_id": 2, "lane": 2,
             "finish_position": 2},
        ]
    )
    out = ft.build(df).sort(["racer_id", "race_date"])
    first = out.filter(pl.col("racer_id") == 1)
    second = out.filter(pl.col("racer_id") == 2)

    assert first["racer_win_raw"].to_list()[1] == pytest.approx(1.0)
    assert second["racer_win_raw"].to_list()[1] == pytest.approx(0.0)


def test_motor_history_is_per_stadium():
    """Motor 10 at stadium 1 and motor 10 at stadium 2 are different motors."""
    df = make_races(
        [
            {"race_date": date(2020, 1, 1), "stadium_id": 1, "motor_no": 10,
             "finish_position": 1},
            {"race_date": date(2020, 1, 2), "race_no": 2, "stadium_id": 2, "motor_no": 10,
             "finish_position": 1},
            {"race_date": date(2020, 1, 3), "race_no": 3, "stadium_id": 2, "motor_no": 10,
             "finish_position": 1},
        ]
    )
    out = ft.build(df).sort("race_date")
    assert out["motor_hist_n"].to_list() == [0, 0, 1]


# ---------------------------------------------------------------------------
# Same-day ordering: the deadline decides what counts as "before"
# ---------------------------------------------------------------------------


def test_same_day_races_are_ordered_by_deadline():
    df = make_races(
        [
            {"race_no": 1, "deadline_time": time(11, 0), "finish_position": 1},
            {"race_no": 2, "deadline_time": time(15, 0), "finish_position": 6},
            {"race_no": 3, "deadline_time": time(13, 0), "finish_position": 6},
        ]
    )
    out = ft.build(df)
    assert out["race_no"].to_list() == [1, 3, 2]
    # By the 15:00 race the racer has two prior starts, one of them a win.
    assert out["racer_win_n"].to_list() == [0, 1, 2]
    assert out["racer_win_raw"].to_list()[2] == pytest.approx(0.5)


def test_missing_deadline_does_not_sort_before_every_race():
    df = make_races(
        [
            {"race_no": 1, "deadline_time": time(10, 0), "finish_position": 1},
            {"race_no": 2, "deadline_time": None, "finish_position": 6},
        ]
    )
    out = ft.build(df)
    assert out["race_no"].to_list() == [1, 2], "null deadline must fall back to midday"


# ---------------------------------------------------------------------------
# Shrinkage (SPEC §2 B: never feed the raw low-denominator rate)
# ---------------------------------------------------------------------------


def target_rows(out: pl.DataFrame) -> pl.DataFrame:
    return out.filter(pl.col("racer_id") == 101).sort("race_date")


def test_shrunk_rate_is_pulled_toward_the_population_mean():
    """A perfect 2-from-2 record must not be reported as a 100% win rate."""
    out = target_rows(ft.build(six_boat_days(3)))

    assert out["racer_win_raw"].to_list()[-1] == pytest.approx(1.0)
    shrunk = out["racer_win_rate"].to_list()[-1]
    assert shrunk < 1.0
    assert shrunk > 1 / 6, "must still sit above the population base rate"


def test_shrinkage_weakens_as_evidence_accumulates():
    few = target_rows(ft.build(six_boat_days(3)))["racer_win_rate"].to_list()[-1]
    many = target_rows(ft.build(six_boat_days(200)))["racer_win_rate"].to_list()[-1]
    assert many > few, "more starts must move the estimate toward the observed rate"


def test_motor_shrinkage_needs_more_evidence_than_racer_shrinkage():
    """SPEC §2 B singles out motors: their denominators are the smallest."""
    assert ft.MOTOR_SHRINKAGE > ft.RACER_SHRINKAGE


def test_shrunk_rate_is_never_null_after_the_first_rows():
    out = ft.build(one_racer_sequence([1, 2, 3, 4]))
    assert out["racer_win_rate"].to_list()[-1] is not None


# ---------------------------------------------------------------------------
# Null handling
# ---------------------------------------------------------------------------


def test_null_start_timing_is_not_counted_as_zero():
    df = make_races(
        [
            {"race_date": date(2020, 1, 1), "start_timing": 0.20},
            {"race_date": date(2020, 1, 2), "race_no": 2, "start_timing": None},
            {"race_date": date(2020, 1, 3), "race_no": 3, "start_timing": 0.20},
        ]
    )
    out = ft.build(df).sort("race_date")
    # Only the two real observations count, so the mean stays 0.20.
    assert out["racer_st_n"].to_list() == [0, 1, 1]
    assert out["racer_st_raw"].to_list()[2] == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# Row-count preservation (Phase 2 completion condition)
# ---------------------------------------------------------------------------


def test_build_preserves_row_count():
    df = one_racer_sequence([1, 2, 3, 4, 5, 6])
    assert ft.build(df).height == df.height


def test_build_handles_an_empty_frame():
    empty = one_racer_sequence([1]).clear()
    assert ft.build(empty).height == 0


# ---------------------------------------------------------------------------
# Feature-set separation: the realised course must not reach the bettable set
# ---------------------------------------------------------------------------


def test_pre_race_features_exclude_the_realised_course():
    for column in ("course", "stadium_course", "wind_x_course"):
        assert column not in ft.PRE_RACE_FEATURES, (
            f"{column!r} is not known when betting closes"
        )


def test_pre_race_features_exclude_outcome_columns():
    banned = {
        "finish_position", "finish_status", "won", "top2", "top3",
        "start_timing", "race_time_sec", "decision", "_course_gain",
    }
    assert banned.isdisjoint(ft.PRE_RACE_FEATURES)


def test_realised_feature_set_is_a_superset():
    assert set(ft.PRE_RACE_FEATURES) < set(ft.REALISED_FEATURES)


def test_feature_columns_switch_respects_the_flag():
    assert "course" not in ft.feature_columns()
    assert "course" in ft.feature_columns(use_realised_course=True)


def test_every_declared_feature_actually_exists():
    out = ft.build(one_racer_sequence([1, 2, 3]))
    missing = [c for c in ft.REALISED_FEATURES if c not in out.columns]
    assert missing == []


# ---------------------------------------------------------------------------
# Encodings are stable across subsets (train/valid/test consistency)
# ---------------------------------------------------------------------------


def test_categorical_encodings_do_not_depend_on_which_rows_are_present():
    both = make_races(
        [
            {"race_no": 1, "weather": "晴", "wind_direction": "北", "grade": "A1"},
            {"race_no": 2, "weather": "雨", "wind_direction": "南西", "grade": "B2"},
        ]
    )
    only_second = both.filter(pl.col("race_no") == 2)

    full = ft.build(both).filter(pl.col("race_no") == 2)
    subset = ft.build(only_second)

    for column in ("weather_code", "wind_dir_code", "grade_code"):
        assert full[column].to_list() == subset[column].to_list(), (
            f"{column!r} depends on the rest of the dataset"
        )


def test_unknown_categories_become_null_not_a_crash():
    df = make_races([{"weather": "みぞれ", "wind_direction": "北北北", "grade": "A3"}])
    out = ft.build(df)
    assert out["weather_code"][0] is None
    assert out["wind_dir_code"][0] is None
    assert out["grade_code"][0] is None


# ---------------------------------------------------------------------------
# Interaction terms (SPEC §2 C)
# ---------------------------------------------------------------------------


def test_wind_interaction_is_zero_when_there_is_no_wind():
    out = ft.build(make_races([{"wind_speed_m": 0, "lane": 3}]))
    assert out["wind_x_lane"][0] == 0.0


def test_head_wind_interaction_only_fires_for_lane_one():
    df = make_races(
        [
            {"race_no": 1, "lane": 1, "wind_direction": "北", "wind_speed_m": 6},
            {"race_no": 2, "lane": 4, "wind_direction": "北", "wind_speed_m": 6},
        ]
    )
    out = ft.build(df).sort("race_no")
    assert out["head_wind_x_lane1"].to_list() == [6.0, 0.0]


def test_stadium_lane_interaction_is_distinct_per_stadium():
    df = make_races(
        [
            {"race_no": 1, "stadium_id": 1, "lane": 2},
            {"race_no": 2, "stadium_id": 12, "lane": 2},
        ]
    )
    out = ft.build(df).sort("race_no")
    assert out["stadium_lane"].to_list() == [12, 122]


def test_course_gain_history_reflects_going_inside():
    """A racer who habitually starts inside their lane must show it in history."""
    df = make_races(
        [
            {"race_date": date(2020, 1, d), "race_no": d, "lane": 4, "course": 1}
            for d in (1, 2, 3)
        ]
    )
    out = ft.build(df).sort("race_date")
    assert out["racer_goes_inside_raw"].to_list()[2] == pytest.approx(1.0)
    assert out["racer_course_gain_raw"].to_list()[2] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Labels for boats that did not finish
# ---------------------------------------------------------------------------


def test_disqualified_boat_is_labelled_as_not_winning():
    """A null finish (F / 転覆 / 失格) must be 0, never null.

    A null label reaches LightGBM as NaN, and it would also vanish from the
    racer's own denominator, inflating the win rate of anyone who is often
    disqualified.
    """
    df = make_races([{"finish_position": None, "finish_status": "F"}])
    out = ft.build(df)
    assert out["won"].to_list() == [0]
    assert out["top2"].to_list() == [0]
    assert out["top3"].to_list() == [0]


def test_no_label_is_null_even_with_missing_finishes():
    df = make_races(
        [
            {"race_date": BASE_DAY, "finish_position": 1},
            {"race_date": BASE_DAY + timedelta(days=1), "finish_position": None},
            {"race_date": BASE_DAY + timedelta(days=2), "finish_position": 4},
        ]
    )
    out = ft.build(df)
    for column in ("won", "top2", "top3"):
        assert out[column].null_count() == 0


def test_a_did_not_finish_start_still_counts_in_the_denominator():
    df = make_races(
        [
            {"race_date": BASE_DAY, "finish_position": 1},
            {"race_date": BASE_DAY + timedelta(days=1), "finish_position": None},
            {"race_date": BASE_DAY + timedelta(days=2), "finish_position": 1},
        ]
    )
    out = ft.build(df).sort("race_date")
    assert out["racer_win_n"].to_list() == [0, 1, 2]
    # One win from two starts, not one win from one start.
    assert out["racer_win_raw"].to_list()[2] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Series grade
# ---------------------------------------------------------------------------


def test_series_grade_does_not_collide_with_the_racer_class():
    """`grade` in entries is the racer's class; the series grade is separate.
    Encoding both under one name would silently null out the racer class."""
    df = make_races([{"race_no": 1}]).with_columns(
        pl.Series("series_grade", ["SG"])
    )
    out = ft.build(df)
    assert out["grade_code"].to_list() == [0], "A1 must still encode as 0"
    assert out["series_grade_rank"].to_list() == [0], "SG is the top series grade"


def test_grade_rank_is_always_created_even_without_a_grade_column():
    """A feature that silently disappears between training and inference is the
    mismatch that only surfaces in production, so the column is unconditional."""
    out = ft.build(one_racer_sequence([1, 2]))
    assert "series_grade_rank" in out.columns
    assert out["series_grade_rank"].to_list() == [None, None]


def test_grade_rank_encodes_the_published_grade():
    from kyotei.schedule import GRADE_RANK

    df = make_races([{"race_no": 1}, {"race_no": 2}]).with_columns(
        pl.Series("series_grade", ["SG", "一般"])
    )
    out = ft.build(df).sort("race_no")
    assert out["series_grade_rank"].to_list() == [GRADE_RANK["SG"], GRADE_RANK["一般"]]


def test_grade_rank_orders_bigger_meetings_lower():
    df = make_races([{"race_no": n} for n in (1, 2, 3)]).with_columns(
        pl.Series("series_grade", ["SG", "G1", "一般"])
    )
    ranks = ft.build(df).sort("race_no")["series_grade_rank"].to_list()
    assert ranks == sorted(ranks), "rank must increase as the meeting gets smaller"


def test_unknown_grade_becomes_null_not_a_crash():
    df = make_races([{"race_no": 1}]).with_columns(pl.Series("series_grade", ["G9"]))
    assert ft.build(df)["series_grade_rank"].to_list() == [None]


def test_grade_is_in_the_morning_feature_set():
    """It is published in advance, so a morning run genuinely has it."""
    assert "series_grade_rank" in ft.MORNING_FEATURES


def test_grade_is_not_an_outcome_column():
    out = ft.build(
        make_races([{"race_no": 1}]).with_columns(pl.Series("series_grade", ["SG"]))
    )
    # Editing the finish must not change the grade encoding.
    tampered = ft.build(
        make_races([{"race_no": 1, "finish_position": 6}]).with_columns(
            pl.Series("series_grade", ["SG"])
        )
    )
    assert out["series_grade_rank"].to_list() == tampered["series_grade_rank"].to_list()
