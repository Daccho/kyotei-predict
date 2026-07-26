"""Tests for the live odds and 直前情報 parsers.

Run against saved real pages (tests/fixtures/), so they are offline and stable.

The strongest check here is not a hand-copied number: it is that the implied
market probabilities, divided by the 75% payback rate, sum to ~1.0. A misparsed
grid -- a shifted column, a dropped block, a confused second/third place --
cannot produce that.
"""

from __future__ import annotations

from datetime import date
from itertools import permutations
from pathlib import Path

import pytest

from kyotei import scrape as sc

FIXTURES = Path(__file__).parent / "fixtures"
ODDS_PAGE = FIXTURES / "odds3t_24_4R_20260725.html"
BEFORE_PAGE = FIXTURES / "beforeinfo_24_4R_20260725.html"


@pytest.fixture(scope="module")
def odds() -> dict[str, float]:
    return sc.parse_trifecta_odds(ODDS_PAGE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def before() -> sc.BeforeInfo:
    return sc.parse_beforeinfo(
        BEFORE_PAGE.read_text(encoding="utf-8"), date(2026, 7, 25), 24, 4
    )


# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------


def test_odds_url_zero_pads_the_stadium():
    url = sc.odds3t_url(date(2026, 7, 25), 4, 12)
    assert "jcd=04" in url and "rno=12" in url and "hd=20260725" in url


def test_beforeinfo_url_shape():
    url = sc.beforeinfo_url(date(2020, 1, 1), 24, 1)
    assert url.endswith("beforeinfo?rno=1&jcd=24&hd=20200101")


# ---------------------------------------------------------------------------
# Odds grid
# ---------------------------------------------------------------------------


def test_all_120_combinations_are_parsed(odds):
    assert len(odds) == 120


def test_no_combination_is_missing(odds):
    expected = {f"{a}-{b}-{c}" for a, b, c in permutations(range(1, 7), 3)}
    assert set(odds) == expected


def test_market_probabilities_sum_to_about_one(odds):
    """The decisive check: 1/odds scaled by the payback rate must total ~1.

    A shifted column or a mixed-up second/third place would not land here.
    """
    total = sum(sc.odds_to_market_probabilities(odds).values())
    assert 0.97 <= total <= 1.03, total


def test_raw_reciprocals_reveal_the_takeout(odds):
    """Without the payback scaling the reciprocals overshoot by ~1/0.75."""
    raw = sum(1.0 / price for price in odds.values())
    assert 1.25 <= raw <= 1.40, raw


def test_specific_odds_match_the_page(odds):
    # Read off the rendered page: 1-2-3 is 33.1, 1-2-4 is 79.3, 3-1-2 is 38.2.
    assert odds["1-2-3"] == pytest.approx(33.1)
    assert odds["1-2-4"] == pytest.approx(79.3)
    assert odds["3-1-2"] == pytest.approx(38.2)


def test_second_and_third_place_are_not_transposed(odds):
    """1-2-3 and 1-3-2 are different tickets and must carry different prices."""
    assert odds["1-2-3"] != odds["1-3-2"]
    assert odds["1-3-2"] == pytest.approx(34.1)


def test_every_price_is_a_plausible_decimal_odd(odds):
    assert all(1.0 <= price <= 100_000 for price in odds.values())


def test_the_favourite_is_cheaper_than_the_longshot(odds):
    assert min(odds.values()) < max(odds.values()) / 10


def test_odds_parser_rejects_a_page_with_no_race():
    with pytest.raises(sc.PageUnavailable):
        sc.parse_trifecta_odds("<html><body>指定されたレースは存在しません</body></html>")


def test_odds_parser_rejects_an_unrelated_page():
    with pytest.raises(sc.PageUnavailable):
        sc.parse_trifecta_odds("<html><body><table><tr><td>x</td></tr></table></body></html>")


def test_market_probability_ignores_zero_prices():
    assert sc.odds_to_market_probabilities({"1-2-3": 0.0}) == {}


def test_market_probability_is_payback_over_price():
    assert sc.odds_to_market_probabilities({"1-2-3": 7.5})["1-2-3"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# 直前情報
# ---------------------------------------------------------------------------


def test_all_six_exhibition_times_are_read(before):
    assert sorted(before.exhibition_time) == [1, 2, 3, 4, 5, 6]
    assert before.complete


def test_exhibition_times_match_the_page(before):
    assert before.exhibition_time == {
        1: 6.99, 2: 6.99, 3: 6.92, 4: 6.97, 5: 6.88, 6: 6.97
    }


def test_exhibition_times_are_in_a_plausible_band(before):
    assert all(6.0 <= t <= 8.0 for t in before.exhibition_time.values())


def test_tilt_is_read_and_can_be_negative(before):
    assert before.tilt[1] == pytest.approx(-0.5)
    assert before.tilt[2] == pytest.approx(0.0)


def test_race_day_weight_is_read(before):
    assert before.weight_kg[1] == pytest.approx(56.9)
    assert all(40 <= w <= 90 for w in before.weight_kg.values())


def test_air_and_water_temperature_are_available_here_unlike_the_k_feed(before):
    """The K feed's weather line stops at 波高; these two only exist on this page."""
    assert before.air_temp_c == pytest.approx(31.0)
    assert before.water_temp_c == pytest.approx(33.0)


def test_wind_and_wave_and_weather_are_read(before):
    assert before.wind_speed_m == pytest.approx(2.0)
    assert before.wave_height_cm == pytest.approx(1.0)
    assert before.weather == "晴"


def test_exhibition_course_is_a_permutation(before):
    courses = before.exhibition_course
    assert sorted(courses) == [1, 2, 3, 4, 5, 6]
    assert sorted(courses.values()) == [1, 2, 3, 4, 5, 6]


def test_exhibition_start_timings_are_plausible(before):
    assert sorted(before.exhibition_start) == [1, 2, 3, 4, 5, 6]
    assert all(0.0 <= st < 1.0 for st in before.exhibition_start.values())


def test_exhibition_start_matches_the_page(before):
    assert before.exhibition_start[2] == pytest.approx(0.04)
    assert before.exhibition_start[4] == pytest.approx(0.28)


def test_beforeinfo_keeps_its_race_identity(before):
    assert (before.race_date, before.stadium_id, before.race_no) == (
        date(2026, 7, 25), 24, 4
    )


def test_beforeinfo_rejects_a_page_with_no_race():
    with pytest.raises(sc.PageUnavailable):
        sc.parse_beforeinfo(
            "<html><body>指定されたレースは存在しません</body></html>",
            date(2026, 7, 25), 24, 4,
        )


def test_beforeinfo_of_an_empty_page_is_simply_incomplete():
    info = sc.parse_beforeinfo("<html><body></body></html>", date(2026, 7, 25), 24, 4)
    assert not info.complete
    assert info.exhibition_time == {}


# ---------------------------------------------------------------------------
# Fetching (no network: an injected fake session)
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status_code: int, body: str = "") -> None:
        self.status_code = status_code
        self.content = body.encode("utf-8")
        self.text = body


class FakeSession:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.requested: list[str] = []

    def get(self, url: str, timeout: int = 0) -> FakeResponse:
        self.requested.append(url)
        return self.response


def test_fetch_raises_on_a_non_200():
    session = FakeSession(FakeResponse(503))
    with pytest.raises(sc.PageUnavailable, match="HTTP 503"):
        sc.fetch_page(session, "https://example.invalid")


def test_fetch_sleeps_before_requesting():
    calls: list[str] = []
    session = FakeSession(FakeResponse(200, "ok"))
    sc.fetch_page(session, "https://example.invalid", sleeper=lambda: calls.append("s"))
    assert calls == ["s"], "the politeness budget applies to these pages too"


def test_fetch_trifecta_odds_parses_a_real_page():
    session = FakeSession(FakeResponse(200, ODDS_PAGE.read_text(encoding="utf-8")))
    parsed = sc.fetch_trifecta_odds(session, date(2026, 7, 25), 24, 4)
    assert len(parsed) == 120
    assert "odds3t?rno=4&jcd=24&hd=20260725" in session.requested[0]


def test_fetch_beforeinfo_parses_a_real_page():
    session = FakeSession(FakeResponse(200, BEFORE_PAGE.read_text(encoding="utf-8")))
    info = sc.fetch_beforeinfo(session, date(2026, 7, 25), 24, 4)
    assert info.complete
    assert info.water_temp_c == pytest.approx(33.0)
