"""Tests for the race-grade scraper, against a saved real monthly page."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from kyotei import schedule as sch

FIXTURE = Path(__file__).parent / "fixtures" / "monthlyschedule_202407.html"


@pytest.fixture(scope="module")
def parsed() -> list[sch.SeriesDay]:
    return sch.parse_monthly_schedule(FIXTURE.read_text(encoding="utf-8"), 2024, 7)


@pytest.fixture(scope="module")
def frame(parsed) -> pl.DataFrame:
    return sch.to_frame(parsed)


def test_url_zero_pads_the_month():
    assert sch.monthly_url(2024, 7).endswith("ym=202407")


def test_all_24_stadiums_appear(frame):
    assert sorted(frame["stadium_id"].unique().to_list()) == list(range(1, 25))


def test_dates_cover_the_month_and_its_leading_days(frame):
    assert frame["race_date"].min() == date(2024, 6, 27)
    assert frame["race_date"].max() == date(2024, 7, 31)


def test_leading_days_are_dated_to_the_previous_month(frame):
    """The grid starts before the 1st; those columns must not be dated to July."""
    june = frame.filter(pl.col("race_date") < date(2024, 7, 1))
    assert june.height > 0
    assert june["race_date"].unique().to_list() == sorted(
        {date(2024, 6, d) for d in range(27, 31)}
    )


def test_every_published_grade_is_recognised(frame):
    assert set(frame["grade"].unique().to_list()) <= set(sch.GRADES.values())


def test_the_ordinary_meeting_dominates(frame):
    counts = dict(
        frame.group_by("grade").agg(pl.len().alias("n")).iter_rows()
    )
    assert counts["一般"] > sum(v for k, v in counts.items() if k != "一般")


def test_the_big_meetings_are_present(frame):
    grades = set(frame["grade"].unique().to_list())
    assert {"SG", "G2", "G3"} <= grades


def test_a_multi_day_series_spans_consecutive_days_at_its_own_stadium(frame):
    """colspan is a day count, so a series occupies a run of consecutive days.

    Checked per stadium, not globally: July 2024 holds two separate SG meetings
    at different venues (one ending 6/30, another starting 7/23), so the SG days
    are two runs rather than one.
    """
    sg = frame.filter(pl.col("grade") == "SG")
    assert sg.height >= 4
    stadiums = sg["stadium_id"].unique().to_list()
    assert len(stadiums) >= 1
    for stadium in stadiums:
        days = sorted(sg.filter(pl.col("stadium_id") == stadium)["race_date"].to_list())
        assert (days[-1] - days[0]).days == len(days) - 1, (
            f"stadium {stadium}: SG days are not consecutive: {days}"
        )


def test_one_series_per_stadium_per_day(frame):
    assert not frame.select(["race_date", "stadium_id"]).is_duplicated().any()


def test_grade_rank_orders_sg_above_ippan():
    assert sch.GRADE_RANK["SG"] < sch.GRADE_RANK["G1"] < sch.GRADE_RANK["一般"]


def test_grade_rank_covers_every_grade():
    assert set(sch.GRADE_RANK) == set(sch.GRADES.values())


def test_parsing_an_empty_page_yields_nothing():
    assert sch.parse_monthly_schedule("<html><body></body></html>", 2024, 7) == []


def test_to_frame_of_nothing_has_the_schema():
    empty = sch.to_frame([])
    assert empty.is_empty()
    assert set(empty.columns) == {"race_date", "stadium_id", "grade", "series"}


# ---------------------------------------------------------------------------
# Joining onto races
# ---------------------------------------------------------------------------


def races_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "race_date": [date(2024, 7, 1)] * 3,
            "stadium_id": pl.Series([24, 24, 13], dtype=pl.Int16),
            "race_no": [1, 2, 1],
        }
    )


def test_attach_grade_preserves_the_row_count(frame):
    out = sch.attach_grade(races_frame(), frame)
    assert out.height == 3


def test_attach_grade_gives_every_race_at_a_stadium_the_same_grade(frame):
    out = sch.attach_grade(races_frame(), frame).filter(pl.col("stadium_id") == 24)
    assert out["grade"].n_unique() == 1


def test_attach_grade_leaves_unknown_days_null(frame):
    unknown = pl.DataFrame(
        {
            "race_date": [date(2030, 1, 1)],
            "stadium_id": pl.Series([24], dtype=pl.Int16),
            "race_no": [1],
        }
    )
    assert sch.attach_grade(unknown, frame)["grade"].to_list() == [None]


def test_attach_grade_rejects_a_schedule_with_duplicate_days(frame):
    duplicated = pl.concat([frame.head(1), frame.head(1)])
    with pytest.raises(ValueError, match="changed the row count"):
        sch.attach_grade(races_frame().head(1).with_columns(
            pl.lit(duplicated["race_date"][0]).alias("race_date"),
            pl.lit(duplicated["stadium_id"][0]).cast(pl.Int16).alias("stadium_id"),
        ), duplicated)


# ---------------------------------------------------------------------------
# Month enumeration
# ---------------------------------------------------------------------------


def test_months_is_inclusive_at_both_ends():
    got = sch.months(date(2024, 11, 5), date(2025, 2, 20))
    assert got == [(2024, 11), (2024, 12), (2025, 1), (2025, 2)]


def test_months_of_a_single_month():
    assert sch.months(date(2024, 7, 1), date(2024, 7, 31)) == [(2024, 7)]


def test_months_for_the_whole_project_span():
    got = sch.months(date(2015, 1, 1), date(2026, 7, 26))
    assert got[0] == (2015, 1) and got[-1] == (2026, 7)
    assert len(got) == 11 * 12 + 7
