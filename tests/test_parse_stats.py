"""Tests for the by-year parse reporting required by Phase 1."""

from __future__ import annotations

from datetime import date

import pytest

from kyotei.parse_stats import FileOutcome, ParseStats


def outcome(day: date | None, ok: int, failed: int = 0, fatal: str | None = None) -> FileOutcome:
    return FileOutcome(
        source=f"b{day:%y%m%d}" if day else "fan1410",
        day=day,
        records_ok=ok,
        records_failed=failed,
        fatal=fatal,
    )


def test_records_are_grouped_by_year():
    stats = ParseStats("B")
    stats.add(outcome(date(2019, 5, 1), 100))
    stats.add(outcome(date(2019, 6, 1), 50))
    stats.add(outcome(date(2020, 1, 1), 70))

    by_year = stats.by_year()
    assert by_year[2019].records_ok == 150
    assert by_year[2019].files == 2
    assert by_year[2020].records_ok == 70


def test_years_are_reported_in_chronological_order():
    stats = ParseStats()
    for year in (2021, 2015, 2018):
        stats.add(outcome(date(year, 3, 3), 10))
    assert list(stats.by_year()) == [2015, 2018, 2021]


def test_success_rate_is_per_year_not_diluted_by_good_years():
    """A layout change in one year must stay visible in that year's rate."""
    stats = ParseStats()
    stats.add(outcome(date(2015, 1, 1), 1000, 0))
    stats.add(outcome(date(2016, 1, 1), 500, 500))  # broken year

    by_year = stats.by_year()
    assert by_year[2015].record_success_rate == 1.0
    assert by_year[2016].record_success_rate == 0.5
    assert stats.total.record_success_rate == pytest.approx(1500 / 2000)


def test_layout_change_suspects_names_the_bad_year_only():
    stats = ParseStats()
    stats.add(outcome(date(2015, 1, 1), 1000))
    stats.add(outcome(date(2016, 1, 1), 900, 100))
    stats.add(outcome(date(2017, 1, 1), 1000))

    assert stats.layout_change_suspects() == [2016]


def test_a_fully_clean_run_has_no_suspects():
    stats = ParseStats()
    stats.add(outcome(date(2015, 1, 1), 1000))
    assert stats.layout_change_suspects() == []


def test_threshold_is_tunable():
    stats = ParseStats()
    stats.add(outcome(date(2015, 1, 1), 9995, 5))  # 99.95%
    assert stats.layout_change_suspects(threshold=0.99) == []
    assert stats.layout_change_suspects(threshold=0.999) == []
    assert stats.layout_change_suspects(threshold=0.9999) == [2015]


def test_worst_year_identifies_the_weakest_year():
    stats = ParseStats()
    stats.add(outcome(date(2015, 1, 1), 100))
    stats.add(outcome(date(2016, 1, 1), 60, 40))
    year, stat = stats.worst_year()
    assert year == 2016
    assert stat.record_success_rate == pytest.approx(0.6)


def test_worst_year_is_none_when_nothing_was_parsed():
    assert ParseStats().worst_year() is None


def test_fatal_file_counts_against_file_success_rate():
    stats = ParseStats()
    stats.add(outcome(date(2015, 1, 1), 100))
    stats.add(outcome(date(2015, 1, 2), 0, fatal="ExtractError: corrupt"))

    stat = stats.by_year()[2015]
    assert stat.files == 2
    assert stat.files_fatal == 1
    assert stat.file_success_rate == 0.5


def test_fatal_reason_is_surfaced_as_a_sample():
    stats = ParseStats()
    stats.add(outcome(date(2015, 1, 2), 0, fatal="ExtractError: corrupt"))
    assert any("corrupt" in s for s in stats.by_year()[2015].samples)


def test_samples_are_capped_per_year():
    stats = ParseStats()
    for day in range(1, 21):
        stats.add(outcome(date(2015, 1, day), 0, fatal="boom"))
    assert len(stats.by_year()[2015].samples) == 5


def test_files_without_a_date_are_grouped_under_none():
    stats = ParseStats()
    stats.add(outcome(None, 1500))
    assert list(stats.by_year()) == [None]


def test_undated_years_sort_last():
    stats = ParseStats()
    stats.add(outcome(None, 10))
    stats.add(outcome(date(2015, 1, 1), 10))
    assert list(stats.by_year()) == [2015, None]


def test_empty_stats_render_without_crashing():
    text = ParseStats("empty").render()
    assert "ALL" in text


def test_render_includes_each_year_and_the_warning():
    stats = ParseStats("K")
    stats.add(outcome(date(2015, 1, 1), 1000))
    stats.add(outcome(date(2016, 1, 1), 500, 500))

    text = stats.render()
    assert "2015" in text and "2016" in text
    assert "ALL" in text
    assert "possible layout change" in text
    assert "50.000%" in text


def test_render_omits_warning_when_all_years_clean():
    stats = ParseStats()
    stats.add(outcome(date(2015, 1, 1), 10))
    assert "possible layout change" not in stats.render()
