"""Tests for the parquet export.

The chunking here exists for a memory reason, so the test that matters is that
chunking changes nothing about the output.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from kyotei import export as ex
from kyotei.paths import raw_path

PILOT = [date(2020, 1, 1), date(2020, 1, 2)]


def available() -> list[date]:
    return [d for d in PILOT if raw_path("B", d).exists() and raw_path("K", d).exists()]


def test_frame_declares_dtypes_rather_than_inferring():
    """finish_status is null for thousands of rows before the first 'K0', which
    an inferred numeric builder then refuses to accept."""
    rows = [{c: None for c in ex.ENTRY_COLUMNS} for _ in range(3)]
    rows[2]["finish_status"] = "K0"
    frame = ex._frame(rows, ex.ENTRY_COLUMNS)
    assert frame.schema["finish_status"] == pl.Utf8
    assert frame["finish_status"].to_list()[2] == "K0"


def test_frame_of_no_rows_still_has_the_schema():
    frame = ex._frame([], ex.ENTRY_COLUMNS)
    assert frame.is_empty()
    assert list(frame.columns) == list(ex.ENTRY_COLUMNS)


def test_join_entries_preserves_row_count():
    races = pl.DataFrame(
        {"race_date": [date(2020, 1, 1)], "stadium_id": [1], "race_no": [1],
         "distance_m": [1800], "has_b": [True], "has_k": [True]}
    )
    entries = pl.DataFrame(
        {"race_date": [date(2020, 1, 1)] * 6, "stadium_id": [1] * 6,
         "race_no": [1] * 6, "lane": list(range(1, 7))}
    )
    joined = ex.join_entries(races, entries)
    assert joined.height == 6


def test_join_entries_rejects_a_fan_out():
    """A duplicated race row would multiply every entry: caught, not shipped."""
    races = pl.DataFrame(
        {"race_date": [date(2020, 1, 1)] * 2, "stadium_id": [1] * 2,
         "race_no": [1] * 2, "distance_m": [1800, 1800],
         "has_b": [True, True], "has_k": [True, True]}
    )
    entries = pl.DataFrame(
        {"race_date": [date(2020, 1, 1)], "stadium_id": [1], "race_no": [1], "lane": [1]}
    )
    with pytest.raises(ValueError, match="changed the row count"):
        ex.join_entries(races, entries)


# ---------------------------------------------------------------------------
# Chunking must not change the result
# ---------------------------------------------------------------------------


def test_monthly_chunking_spans_a_month_boundary_without_losing_rows():
    days = available()
    if len(days) < 2:
        pytest.skip("needs two consecutive pilot days on disk")
    # 2019-12-31 .. 2020-01-02 crosses a month and a year boundary; the flush
    # happens at the boundary, so any off-by-one would drop or duplicate a day.
    races, entries, payouts, b_stats, _ = ex.parse_range(
        date(2019, 12, 31), days[-1], progress_every=0
    )
    per_day = {}
    for day in (date(2019, 12, 31), *days):
        _, day_entries, _, _, _ = ex.parse_day(day)
        per_day[day] = len(day_entries)
    assert entries.height == sum(per_day.values())


def test_parse_range_reports_stats_per_year():
    days = available()
    if not days:
        pytest.skip("no pilot archives on disk")
    _, _, _, b_stats, k_stats = ex.parse_range(days[0], days[-1], progress_every=0)
    assert set(b_stats.by_year()) == {2020}
    assert b_stats.total.records_ok > 0
    assert k_stats.total.records_ok > 0


def test_parse_range_of_an_empty_window_returns_typed_empties():
    races, entries, payouts, _, _ = ex.parse_range(
        date(1990, 1, 1), date(1990, 1, 2), progress_every=0
    )
    assert races.is_empty() and entries.is_empty() and payouts.is_empty()
    assert list(entries.columns) == list(ex.ENTRY_COLUMNS)
