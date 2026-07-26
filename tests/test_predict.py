"""Tests for the daily inference report (Phase 6).

The point of these is the timing contract: a model that needs 直前情報 or the
realised course must not be usable for a morning run, and the report must be
honest when there is nothing worth buying.
"""

from __future__ import annotations

from datetime import date, time

import polars as pl
import pytest

from kyotei import features as ft
from kyotei import model as md
from kyotei import predict as pr


# ---------------------------------------------------------------------------
# Feature-set timing contract
# ---------------------------------------------------------------------------


def test_morning_feature_set_excludes_late_information():
    for column in ("exhibition_time", "wind_speed_m", "weather_code", "wave_x_lane"):
        assert column not in ft.MORNING_FEATURES, (
            f"{column!r} is only published shortly before the deadline"
        )


def test_morning_feature_set_excludes_the_realised_course():
    assert "course" not in ft.MORNING_FEATURES
    assert "stadium_course" not in ft.MORNING_FEATURES


def test_morning_is_a_strict_subset_of_prerace():
    assert set(ft.MORNING_FEATURES) < set(ft.PRE_RACE_FEATURES)


def test_morning_still_carries_the_course_tendency_history():
    """Course is unknown, but a racer's habit of taking an inside course is not."""
    assert "racer_course_gain_rate" in ft.MORNING_FEATURES
    assert "racer_goes_inside_rate" in ft.MORNING_FEATURES


def test_morning_keeps_the_published_form_columns():
    for column in ("lane", "grade_code", "national_win_rate", "motor_top2_rate"):
        assert column in ft.MORNING_FEATURES


def test_feature_columns_rejects_an_unknown_set():
    with pytest.raises(ValueError, match="unknown feature set"):
        ft.feature_columns("tomorrow")


def test_feature_columns_named_sets_match_the_constants():
    assert ft.feature_columns("morning") == ft.MORNING_FEATURES
    assert ft.feature_columns("prerace") == ft.PRE_RACE_FEATURES
    assert ft.feature_columns("realised") == ft.REALISED_FEATURES


def test_realised_shorthand_still_works():
    assert ft.feature_columns(use_realised_course=True) == ft.REALISED_FEATURES


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def scored_frame(probabilities=(0.55, 0.15, 0.10, 0.08, 0.07, 0.05)) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "race_date": [date(2026, 7, 26)] * 6,
            "stadium_id": [24] * 6,
            "race_no": [4] * 6,
            "lane": list(range(1, 7)),
            "p_win": list(probabilities),
        }
    )


def priced_frame(evs: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "race_date": [date(2026, 7, 26)] * len(evs),
            "stadium_id": [24] * len(evs),
            "race_no": [4] * len(evs),
            "combination": [f"1-2-{i}" for i in range(3, 3 + len(evs))],
            "p_model": [0.02] * len(evs),
            "expected_payout": [1500.0] * len(evs),
            "ev": evs,
        }
    )


def test_report_lists_picks_above_the_threshold():
    report = pr.render_report(
        date(2026, 7, 26), scored_frame(), priced_frame([1.5, 0.9]), 1.2, "morning"
    )
    assert "1-2-3" in report
    assert "1-2-4" not in report.split("## 各レースの1着確率")[0]


def test_report_says_so_when_nothing_is_worth_buying():
    report = pr.render_report(
        date(2026, 7, 26), scored_frame(), priced_frame([0.5, 0.6]), 1.2, "morning"
    )
    assert "買い目なし" in report
    assert "見送り" in report, "skipping must be presented as a valid outcome"


def test_report_names_the_stadium_in_japanese():
    report = pr.render_report(
        date(2026, 7, 26), scored_frame(), priced_frame([1.5]), 1.2, "morning"
    )
    assert "大村" in report


def test_report_includes_every_lane_probability():
    report = pr.render_report(
        date(2026, 7, 26), scored_frame(), priced_frame([1.5]), 1.2, "morning"
    )
    section = report.split("## 各レースの1着確率")[1]
    for lane in range(1, 7):
        assert f"{lane}号艇" in section


def test_report_warns_that_prices_are_historical_averages():
    report = pr.render_report(
        date(2026, 7, 26), scored_frame(), priced_frame([1.5]), 1.2, "morning"
    )
    assert "過去平均配当" in report
    assert "オッズ" in report


def test_report_states_the_morning_information_limit():
    report = pr.render_report(
        date(2026, 7, 26), scored_frame(), priced_frame([1.5]), 1.2, "morning"
    )
    assert "直前情報" in report


def test_report_reports_the_purchase_share():
    report = pr.render_report(
        date(2026, 7, 26), scored_frame(), priced_frame([1.5] + [0.1] * 9), 1.2, "morning"
    )
    assert "10 点中" in report or "全 10 点" in report


def test_report_mentions_the_threshold_it_used():
    report = pr.render_report(
        date(2026, 7, 26), scored_frame(), priced_frame([1.5]), 1.35, "morning"
    )
    assert "1.35" in report


def test_report_is_valid_markdown_table_shape():
    report = pr.render_report(
        date(2026, 7, 26), scored_frame(), priced_frame([1.5, 1.4]), 1.2, "morning"
    )
    rows = [line for line in report.splitlines() if line.startswith("| 大村")]
    assert len(rows) == 2
    assert all(line.count("|") == 7 for line in rows)


# ---------------------------------------------------------------------------
# Schema alignment between history and today's card
# ---------------------------------------------------------------------------


def history_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "race_date": [date(2026, 7, 25)],
            "stadium_id": [24],
            "race_no": [1],
            "lane": [1],
            "course": [1],
            "racer_id": [4000],
            "won": [1],
        }
    )


def test_align_adds_missing_columns_and_keeps_order():
    today = pl.DataFrame(
        {
            "race_date": [date(2026, 7, 26)],
            "stadium_id": [24],
            "race_no": [4],
            "lane": [1],
            "racer_id": [4001],
        }
    )
    out = pr.align_to_history(history_frame(), today)
    assert out.columns == history_frame().columns
    assert out.height == 2
    assert out["course"].to_list()[1] is None


def test_align_preserves_history_rows_unchanged():
    history = history_frame()
    today = pl.DataFrame(
        {
            "race_date": [date(2026, 7, 26)],
            "stadium_id": [24],
            "race_no": [4],
            "lane": [2],
            "racer_id": [4002],
        }
    )
    out = pr.align_to_history(history, today)
    assert out.head(1).to_dicts() == history.to_dicts()


def test_stadium_names_cover_all_24():
    assert sorted(pr.STADIUM_NAMES) == list(range(1, 25))
    assert all(isinstance(v, str) and v for v in pr.STADIUM_NAMES.values())


# ---------------------------------------------------------------------------
# Race selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("24", 24),
        ("1", 1),
        ("大村", 24),
        ("ボートレース大村", 24),
        ("大村競艇", 24),
        ("大村競艇場", 24),
        ("  住之江  ", 12),
        ("びわこ", 11),
    ],
)
def test_resolve_stadium_accepts_codes_and_names(given, expected):
    assert pr.resolve_stadium(given) == expected


def test_resolve_stadium_rejects_an_out_of_range_code():
    with pytest.raises(ValueError, match="1-24"):
        pr.resolve_stadium("25")


def test_resolve_stadium_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="unknown stadium"):
        pr.resolve_stadium("架空ボート")


def multi_race_frame() -> pl.DataFrame:
    rows = []
    for stadium in (12, 24):
        for race_no in (3, 4):
            for lane in range(1, 7):
                rows.append(
                    {
                        "race_date": date(2026, 7, 26),
                        "stadium_id": stadium,
                        "race_no": race_no,
                        "lane": lane,
                        "p_win": 1 / 6,
                    }
                )
    return pl.DataFrame(rows)


def test_select_races_narrows_to_one_stadium():
    out = pr.select_races(multi_race_frame(), stadium=24)
    assert out["stadium_id"].unique().to_list() == [24]
    assert out.height == 12


def test_select_races_narrows_to_one_race_number():
    out = pr.select_races(multi_race_frame(), race_no=4)
    assert out["race_no"].unique().to_list() == [4]
    assert out.height == 12


def test_select_races_narrows_to_a_single_race():
    out = pr.select_races(multi_race_frame(), stadium=24, race_no=4)
    assert out.height == 6, "one race is exactly six boats"
    assert out["stadium_id"].unique().to_list() == [24]
    assert out["race_no"].unique().to_list() == [4]


def test_select_races_without_filters_is_a_no_op():
    frame = multi_race_frame()
    assert pr.select_races(frame).height == frame.height


def test_select_races_returns_empty_for_a_race_that_did_not_run():
    assert pr.select_races(multi_race_frame(), stadium=24, race_no=11).is_empty()


def test_selecting_one_race_keeps_its_probabilities_summing_to_one():
    """Narrowing must not renormalise or otherwise disturb the race."""
    out = pr.select_races(multi_race_frame(), stadium=24, race_no=4)
    assert out["p_win"].sum() == pytest.approx(1.0)
