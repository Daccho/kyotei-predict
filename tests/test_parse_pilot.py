"""Cross-validation of the B/K parsers against real archives.

Because the published layout page does not document the B/K feeds (it covers
only fan{YYMM}), the offsets in parse_b/parse_k are transcribed from each
block's own column header. Transcription can be wrong in ways that still parse
cleanly, so correctness is established by agreement between *independent*
sources rather than by the parse succeeding:

  * motor and boat numbers appear in both feeds -- they must agree per lane;
  * the trifecta dividend's combination must equal the lanes of the first three
    finishers, which can only hold if 着, 艇 and the dividend block are all read
    correctly;
  * every race must yield exactly 6 boats;
  * 進入 must be a permutation within a race, and must not merely copy 艇.

Skipped when the raw archives have not been downloaded yet.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from kyotei import parse_b, parse_k
from kyotei.download import extract_bytes
from kyotei.paths import raw_path

#: Days used as fixtures. Chosen only because they are the first days of the
#: pilot year; nothing about them is special.
PILOT_DAYS = [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)]


def available_days() -> list[date]:
    return [d for d in PILOT_DAYS if raw_path("B", d).exists() and raw_path("K", d).exists()]


@pytest.fixture(scope="module")
def parsed() -> list[tuple[date, list, list]]:
    days = available_days()
    if not days:
        pytest.skip("no raw B+K archives on disk; run python -m kyotei.download first")
    out = []
    for day in days:
        b, _ = parse_b.parse_payload(extract_bytes(raw_path("B", day)), day, f"b{day:%y%m%d}")
        k, _ = parse_k.parse_payload(extract_bytes(raw_path("K", day)), day, f"k{day:%y%m%d}")
        out.append((day, b, k))
    return out


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_every_race_has_exactly_six_boats(parsed):
    offenders = [
        (day, r.key, len(r.entries))
        for day, b_races, k_races in parsed
        for r in (*b_races, *k_races)
        if len(r.entries) != 6
    ]
    assert offenders == []


def test_both_feeds_describe_the_same_races(parsed):
    for day, b_races, k_races in parsed:
        b_keys = {r.key for r in b_races}
        k_keys = {r.key for r in k_races}
        assert b_keys == k_keys, f"{day}: B and K disagree on which races ran"


def test_race_numbers_are_in_range_and_unique_per_stadium(parsed):
    for day, b_races, _ in parsed:
        assert len({r.key for r in b_races}) == len(b_races), f"{day}: duplicate race key"
        assert all(1 <= r.race_no <= 12 for r in b_races)


def test_stadium_ids_are_valid(parsed):
    for _, b_races, k_races in parsed:
        for r in (*b_races, *k_races):
            assert 1 <= r.stadium_id <= 24


def test_parsing_yields_no_record_failures(parsed):
    """A clean parse on real data; failures would show up per year in the CLI."""
    for day, _, _ in parsed:
        b, b_outcome = parse_b.parse_payload(
            extract_bytes(raw_path("B", day)), day, f"b{day:%y%m%d}"
        )
        k, k_outcome = parse_k.parse_payload(
            extract_bytes(raw_path("K", day)), day, f"k{day:%y%m%d}"
        )
        assert b_outcome.records_failed == 0, b_outcome.samples
        assert k_outcome.records_failed == 0, k_outcome.samples


# ---------------------------------------------------------------------------
# Agreement between the two independent feeds
# ---------------------------------------------------------------------------


def test_motor_and_boat_numbers_agree_between_b_and_k(parsed):
    """The strongest offset check available: both feeds publish these numbers,
    from different byte positions in differently-shaped records."""
    mismatches = []
    for day, b_races, k_races in parsed:
        k_by_key = {r.key: r for r in k_races}
        for b in b_races:
            k = k_by_key.get(b.key)
            if k is None:
                continue
            b_by_lane = {e["lane"]: e for e in b.entries}
            for entry in k.entries:
                other = b_by_lane.get(entry["lane"])
                if other is None:
                    continue
                if entry["motor_no"] is not None and other["motor_no"] != entry["motor_no"]:
                    mismatches.append((day, b.key, entry["lane"], "motor"))
                if entry["boat_no"] is not None and other["boat_no"] != entry["boat_no"]:
                    mismatches.append((day, b.key, entry["lane"], "boat"))
    assert mismatches == []


def test_racer_ids_agree_between_b_and_k(parsed):
    mismatches = []
    for day, b_races, k_races in parsed:
        k_by_key = {r.key: r for r in k_races}
        for b in b_races:
            k = k_by_key.get(b.key)
            if k is None:
                continue
            b_by_lane = {e["lane"]: e["racer_id"] for e in b.entries}
            for entry in k.entries:
                if entry["racer_id"] is None:
                    continue
                if b_by_lane.get(entry["lane"]) != entry["racer_id"]:
                    mismatches.append((day, b.key, entry["lane"]))
    assert mismatches == []


def test_trifecta_dividend_matches_the_top_three_lanes(parsed):
    """Ties together 着, 艇 and the dividend block in one assertion."""
    checked = 0
    bad = []
    for day, _, k_races in parsed:
        for race in k_races:
            top = {e["finish_position"]: e["lane"] for e in race.entries
                   if e["finish_position"] in (1, 2, 3)}
            trifecta = [p for p in race.payouts if p["bet_type"] == "trifecta"]
            if len(top) != 3 or not trifecta:
                continue
            expected = f"{top[1]}-{top[2]}-{top[3]}"
            if not any(p["combination"] == expected for p in trifecta):
                bad.append((day, race.key, expected, [p["combination"] for p in trifecta]))
            checked += 1
    assert checked > 0, "no race had both a full podium and a trifecta dividend"
    assert bad == []


def test_win_dividend_matches_the_winning_lane(parsed):
    bad = []
    for day, _, k_races in parsed:
        for race in k_races:
            winner = next((e["lane"] for e in race.entries if e["finish_position"] == 1), None)
            win = [p for p in race.payouts if p["bet_type"] == "win"]
            if winner is None or not win:
                continue
            if not any(p["combination"] == str(winner) for p in win):
                bad.append((day, race.key, winner, [p["combination"] for p in win]))
    assert bad == []


# ---------------------------------------------------------------------------
# lane vs course -- the distinction the whole model rests on
# ---------------------------------------------------------------------------


def test_course_is_a_permutation_within_each_race(parsed):
    for day, _, k_races in parsed:
        for race in k_races:
            courses = [e["course"] for e in race.entries if e["course"] is not None]
            assert len(courses) == len(set(courses)), f"{day} {race.key}: duplicate course"
            assert all(1 <= c <= 6 for c in courses)


def test_course_is_not_just_a_copy_of_lane(parsed):
    """If the parser were reading 艇 twice this would be 0 and the dominant
    feature in the whole model would silently be the draw instead."""
    differing = sum(
        1
        for _, _, k_races in parsed
        for race in k_races
        for e in race.entries
        if e["course"] is not None and e["course"] != e["lane"]
    )
    assert differing > 0


def test_course_one_win_rate_is_plausible(parsed):
    """Course 1 wins roughly half the time. A wildly different number means
    course or finish is being read from the wrong offset."""
    wins = total = 0
    for _, _, k_races in parsed:
        for race in k_races:
            for e in race.entries:
                if e["finish_position"] == 1 and e["course"] is not None:
                    total += 1
                    wins += e["course"] == 1
    assert total > 0
    assert 0.40 <= wins / total <= 0.70, f"course-1 win rate {wins / total:.3f}"


# ---------------------------------------------------------------------------
# Field sanity
# ---------------------------------------------------------------------------


def test_grades_are_from_the_known_set(parsed):
    seen = {e["grade"] for _, b_races, _ in parsed for r in b_races for e in r.entries}
    assert seen <= {"A1", "A2", "B1", "B2", None, ""}


def test_racer_names_are_non_empty_and_not_padded(parsed):
    for _, b_races, _ in parsed:
        for race in b_races:
            for entry in race.entries:
                name = entry["racer_name"]
                assert name, f"{race.key} lane {entry['lane']}: empty name"
                assert name == name.strip()
                assert "　" not in name, "full-width padding must be stripped"


def test_short_racer_names_agree_between_b_and_k(parsed):
    """B writes '林　恵祐', K writes '林　恵　祐'; both must normalise alike.

    Only names that fit B's 4-character field are compared. Longer names are
    abbreviated by B in a way that is *not* a prefix of the full name
    ('マイケル田代' becomes 'マイケ田'), which is exactly why nothing in this
    pipeline joins on racer_name -- see the next test.
    """
    mismatches = []
    for day, b_races, k_races in parsed:
        k_by_key = {r.key: r for r in k_races}
        for b in b_races:
            k = k_by_key.get(b.key)
            if k is None:
                continue
            k_names = {e["lane"]: e["racer_name"] for e in k.entries}
            for entry in b.entries:
                other = k_names.get(entry["lane"])
                if not other or not entry["racer_name"] or len(other) > 4:
                    continue
                if other != entry["racer_name"]:
                    mismatches.append((day, b.key, entry["lane"], entry["racer_name"], other))
    assert mismatches[:5] == []


def test_b_abbreviates_long_names_so_only_racer_id_is_a_safe_join_key(parsed):
    """Documents the trap: B's name field is 8 bytes, and long names are
    abbreviated rather than truncated, so name-based joins would silently
    mismatch. racer_id agreement is asserted separately."""
    abbreviated = []
    for _, b_races, k_races in parsed:
        k_by_key = {r.key: r for r in k_races}
        for b in b_races:
            k = k_by_key.get(b.key)
            if k is None:
                continue
            k_names = {e["lane"]: e["racer_name"] for e in k.entries}
            for entry in b.entries:
                other = k_names.get(entry["lane"]) or ""
                if len(other) > 4 and other[:4] != entry["racer_name"]:
                    abbreviated.append((entry["racer_name"], other))
    assert abbreviated, "expected at least one abbreviated long name in the pilot days"
    for short, full in abbreviated:
        assert len(short) <= 4


def test_rates_are_in_plausible_ranges(parsed):
    for _, b_races, _ in parsed:
        for race in b_races:
            for e in race.entries:
                if e["national_win_rate"] is not None:
                    assert 0 <= e["national_win_rate"] <= 10
                if e["national_top2_rate"] is not None:
                    assert 0 <= e["national_top2_rate"] <= 100
                if e["motor_top2_rate"] is not None:
                    assert 0 <= e["motor_top2_rate"] <= 100


def test_deadline_times_are_present_and_ordered_within_a_stadium(parsed):
    for day, b_races, _ in parsed:
        by_stadium: dict[int, list] = {}
        for race in b_races:
            if race.deadline_time:
                by_stadium.setdefault(race.stadium_id, []).append((race.race_no, race.deadline_time))
        assert by_stadium, f"{day}: no deadline times parsed"
        for stadium, items in by_stadium.items():
            ordered = [t for _, t in sorted(items)]
            assert ordered == sorted(ordered), f"{day} stadium {stadium}: deadlines out of order"


def test_distance_is_the_standard_race_distance(parsed):
    distances = {r.distance_m for _, b_races, _ in parsed for r in b_races}
    assert distances <= {1800, 1200, 600, 2600, 3000}, distances


def test_weather_and_wind_are_populated(parsed):
    races = [r for _, _, k_races in parsed for r in k_races]
    assert sum(1 for r in races if r.weather) / len(races) > 0.9
    assert sum(1 for r in races if r.wind_speed_m is not None) / len(races) > 0.9


def test_decision_when_present_is_a_known_move(parsed):
    seen = {r.decision for _, _, k_races in parsed for r in k_races if r.decision}
    assert seen, "決まり手 was never parsed"
    assert seen <= set(parse_k.DECISIONS) | {"不"}, seen


def test_start_timings_are_in_a_plausible_band(parsed):
    values = [
        e["start_timing"]
        for _, _, k_races in parsed
        for r in k_races
        for e in r.entries
        if e["start_timing"] is not None
    ]
    assert values
    assert all(-1.0 < v < 2.0 for v in values), "ST outside plausible range"


def test_race_times_are_plausible_for_1800m(parsed):
    values = [
        e["race_time_sec"]
        for _, _, k_races in parsed
        for r in k_races
        for e in r.entries
        if e["race_time_sec"] is not None
    ]
    assert values
    assert all(90 < v < 180 for v in values), f"outliers: {[v for v in values if not 90 < v < 180][:5]}"


def test_payout_types_are_complete_per_race(parsed):
    """Each finished race publishes all seven dividend types."""
    expected = set(parse_k.BET_LABELS.values())
    for day, _, k_races in parsed:
        for race in k_races:
            if not race.payouts:
                continue
            assert {p["bet_type"] for p in race.payouts} == expected, f"{day} {race.key}"


def test_wide_always_has_three_combinations(parsed):
    for day, _, k_races in parsed:
        for race in k_races:
            if not race.payouts:
                continue
            wide = sum(1 for p in race.payouts if p["bet_type"] == "wide")
            assert wide == 3, f"{day} {race.key}: wide {wide}"


def test_place_has_one_or_two_combinations(parsed):
    """複勝 normally pays the first two finishers, but not always.

    On 2020-01-03 two of 228 races publish a single 複勝 line -- e.g. stadium 5
    race 10, where the raw feed reads '複勝     4          380' with no second
    entry even though all six boats finished and boat 6 placed second. The
    parser reproduces the file faithfully; the reason the feed omits the second
    payout is not documented anywhere we have found, so this asserts the range
    the data actually occupies rather than an invented rule.

    Nothing downstream depends on it: the backtest settles trifecta only.
    """
    seen: set[int] = set()
    for day, _, k_races in parsed:
        for race in k_races:
            if not race.payouts:
                continue
            place = sum(1 for p in race.payouts if p["bet_type"] == "place")
            assert place in (1, 2), f"{day} {race.key}: place {place}"
            seen.add(place)
    assert 2 in seen, "two payouts must still be the normal case"


def test_payouts_are_per_hundred_yen_and_positive(parsed):
    for _, _, k_races in parsed:
        for race in k_races:
            for p in race.payouts:
                assert p["payout_yen"] >= 100 or p["payout_yen"] == 0
                assert p["payout_yen"] < 10_000_000
