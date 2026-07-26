"""Tests for the resumable odds backfill.

The two properties that matter for a ~15-hour job in a container that may not
live that long are tested here: an interrupted run is still an unbiased sample
of the season, and a resumed run does not refetch what it already has.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from kyotei import odds_backfill as ob
from kyotei import scrape as sc

FIXTURE = Path(__file__).parent / "fixtures" / "odds3t_24_4R_20260725.html"


def make_races(days: int = 30, stadiums: int = 4, races_per_day: int = 12):
    return [
        ob.RaceRef(date(2024, 1, 1) + timedelta(days=d), s, r)
        for d in range(days)
        for s in range(1, stadiums + 1)
        for r in range(1, races_per_day + 1)
    ]


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------


def test_race_key_is_stable_and_zero_padded():
    assert ob.RaceRef(date(2024, 3, 4), 7, 12).key() == "2024-03-04/07/12"


def test_race_keys_are_unique_per_race():
    races = make_races(days=5)
    assert len({r.key() for r in races}) == len(races)


# ---------------------------------------------------------------------------
# Visiting order: any prefix must be a fair sample of the season
# ---------------------------------------------------------------------------


def test_visiting_order_is_a_permutation():
    races = make_races()
    ordered = ob.visiting_order(races)
    assert len(ordered) == len(races)
    assert {r.key() for r in ordered} == {r.key() for r in races}


def test_visiting_order_is_deterministic():
    races = make_races()
    first = [r.key() for r in ob.visiting_order(races)]
    second = [r.key() for r in ob.visiting_order(races)]
    assert first == second, "a resumed run must continue the same sequence"


def test_visiting_order_does_not_depend_on_input_order():
    races = make_races(days=10)
    forward = [r.key() for r in ob.visiting_order(races)]
    backward = [r.key() for r in ob.visiting_order(list(reversed(races)))]
    assert forward == backward


def test_visiting_order_is_not_chronological():
    """Chronological order would make an interrupted run January-only."""
    ordered = ob.visiting_order(make_races(days=60))
    days = [r.day for r in ordered]
    assert days != sorted(days)


def test_any_prefix_spans_the_whole_period():
    """The point of shuffling: a 10% prefix must touch most months, not just the
    first one."""
    races = make_races(days=360, stadiums=2, races_per_day=6)
    ordered = ob.visiting_order(races)
    prefix = ordered[: len(ordered) // 10]
    months = {r.day.month for r in prefix}
    assert len(months) == 12, f"prefix only covered months {sorted(months)}"


def test_a_chronological_prefix_would_have_been_biased():
    """Contrast case, so the previous test's value is explicit."""
    races = sorted(make_races(days=360, stadiums=2, races_per_day=6),
                   key=lambda r: (r.day, r.stadium_id, r.race_no))
    prefix = races[: len(races) // 10]
    assert len({r.day.month for r in prefix}) <= 2


def test_prefix_is_roughly_proportional_across_months():
    races = make_races(days=360, stadiums=2, races_per_day=6)
    ordered = ob.visiting_order(races)
    prefix = ordered[: len(ordered) // 4]
    counts = {}
    for ref in prefix:
        counts[ref.day.month] = counts.get(ref.day.month, 0) + 1
    expected = len(prefix) / 12
    assert all(0.6 * expected <= c <= 1.4 * expected for c in counts.values()), counts


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


def test_load_done_on_a_missing_file_is_empty(tmp_path):
    assert ob.load_done(tmp_path / "nothing.jsonl") == set()


def test_writer_appends_one_line_per_record(tmp_path):
    path = tmp_path / "odds.jsonl"
    writer = ob.Writer(path)
    writer.write({"k": "a", "o": {"1-2-3": 1.5}})
    writer.write({"k": "b", "unavailable": "nope"})
    writer.close()
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_load_done_reads_back_written_keys(tmp_path):
    path = tmp_path / "odds.jsonl"
    writer = ob.Writer(path)
    writer.write({"k": "2024-01-01/24/1", "o": {}})
    writer.close()
    assert ob.load_done(path) == {"2024-01-01/24/1"}


def test_unavailable_races_are_remembered_so_they_are_not_retried(tmp_path):
    path = tmp_path / "odds.jsonl"
    writer = ob.Writer(path)
    writer.write({"k": "2024-01-01/24/1", "unavailable": "no odds"})
    writer.close()
    assert "2024-01-01/24/1" in ob.load_done(path)


def test_load_done_survives_a_truncated_final_line(tmp_path):
    """A killed run can leave half a line; that must not crash the resume."""
    path = tmp_path / "odds.jsonl"
    path.write_text(
        json.dumps({"k": "2024-01-01/24/1", "o": {}}) + "\n" + '{"k": "2024-01-0',
        encoding="utf-8",
    )
    assert ob.load_done(path) == {"2024-01-01/24/1"}


# ---------------------------------------------------------------------------
# Backfill behaviour with an injected session
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.content = body.encode("utf-8")
        self.text = body


class FakeSession:
    def __init__(self, body: str, status_code: int = 200) -> None:
        self.body = body
        self.status_code = status_code
        self.requested: list[str] = []
        self._lock = __import__("threading").Lock()

    def get(self, url: str, timeout: int = 0) -> FakeResponse:
        with self._lock:
            self.requested.append(url)
        return FakeResponse(self.status_code, self.body)


def no_wait() -> ob.RateLimiter:
    return ob.RateLimiter(1.0, clock=lambda: 0.0, sleep=lambda _s: None)


@pytest.fixture(scope="module")
def real_page() -> str:
    return FIXTURE.read_text(encoding="utf-8")


def test_backfill_records_every_race(tmp_path, real_page):
    races = make_races(days=1, stadiums=1, races_per_day=3)
    out = tmp_path / "odds.jsonl"
    session = FakeSession(real_page)

    progress = ob.backfill(races, out, session=session, limiter=no_wait(), workers=3)

    assert progress.fetched == 3
    assert len(ob.load_done(out)) == 3


def test_backfill_skips_races_already_recorded(tmp_path, real_page):
    races = make_races(days=1, stadiums=1, races_per_day=3)
    out = tmp_path / "odds.jsonl"
    session = FakeSession(real_page)
    ob.backfill(races, out, session=session, limiter=no_wait(), workers=3)

    second = FakeSession(real_page)
    progress = ob.backfill(races, out, session=second, limiter=no_wait(), workers=3)

    assert second.requested == [], "a resumed run must not refetch"
    assert progress.attempted == 0


def test_backfill_honours_the_limit(tmp_path, real_page):
    races = make_races(days=2, stadiums=2, races_per_day=6)
    out = tmp_path / "odds.jsonl"
    session = FakeSession(real_page)

    progress = ob.backfill(races, out, session=session, limiter=no_wait(),
                           workers=4, limit=5)

    assert progress.attempted == 5
    assert len(session.requested) == 5


def test_backfill_records_unavailable_pages_without_failing(tmp_path):
    races = make_races(days=1, stadiums=1, races_per_day=2)
    out = tmp_path / "odds.jsonl"
    session = FakeSession("<html>指定されたレースは存在しません</html>")

    progress = ob.backfill(races, out, session=session, limiter=no_wait(), workers=2)

    assert progress.unavailable == 2
    assert progress.fetched == 0


def test_backfill_records_http_errors_rather_than_crashing(tmp_path):
    races = make_races(days=1, stadiums=1, races_per_day=2)
    out = tmp_path / "odds.jsonl"
    session = FakeSession("boom", status_code=503)

    progress = ob.backfill(races, out, session=session, limiter=no_wait(), workers=2)

    assert progress.unavailable == 2, "an HTTP error is surfaced as unavailable"
    assert progress.fetched == 0


def test_backfill_requests_the_right_urls(tmp_path, real_page):
    races = [ob.RaceRef(date(2024, 5, 6), 7, 11)]
    session = FakeSession(real_page)
    ob.backfill(races, tmp_path / "o.jsonl", session=session, limiter=no_wait(), workers=1)
    assert "rno=11&jcd=07&hd=20240506" in session.requested[0]


# ---------------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------------


def test_to_frame_of_a_missing_file_has_the_right_schema(tmp_path):
    frame = ob.to_frame(tmp_path / "nothing.jsonl")
    assert frame.is_empty()
    assert set(frame.columns) == {
        "race_date", "stadium_id", "race_no", "combination", "odds"
    }


def test_to_frame_explodes_one_row_per_combination(tmp_path, real_page):
    races = make_races(days=1, stadiums=1, races_per_day=2)
    out = tmp_path / "odds.jsonl"
    ob.backfill(races, out, session=FakeSession(real_page), limiter=no_wait(), workers=2)

    frame = ob.to_frame(out)
    assert frame.height == 2 * 120
    assert frame["combination"].n_unique() == 120


def test_to_frame_keeps_the_race_identity(tmp_path, real_page):
    races = [ob.RaceRef(date(2024, 5, 6), 7, 11)]
    out = tmp_path / "odds.jsonl"
    ob.backfill(races, out, session=FakeSession(real_page), limiter=no_wait(), workers=1)

    frame = ob.to_frame(out)
    assert frame["race_date"].unique().to_list() == [date(2024, 5, 6)]
    assert frame["stadium_id"].unique().to_list() == [7]
    assert frame["race_no"].unique().to_list() == [11]


def test_to_frame_skips_unavailable_records(tmp_path):
    out = tmp_path / "odds.jsonl"
    writer = ob.Writer(out)
    writer.write({"k": "a", "d": "2024-01-01", "j": 1, "r": 1, "unavailable": "x"})
    writer.close()
    assert ob.to_frame(out).is_empty()


def test_odds_round_trip_preserves_prices(tmp_path, real_page):
    expected = sc.parse_trifecta_odds(real_page)
    out = tmp_path / "odds.jsonl"
    ob.backfill([ob.RaceRef(date(2024, 1, 1), 1, 1)], out,
                session=FakeSession(real_page), limiter=no_wait(), workers=1)

    frame = ob.to_frame(out)
    got = dict(zip(frame["combination"], frame["odds"]))
    assert got == pytest.approx(expected)


def test_coverage_reports_months(tmp_path, real_page):
    races = [
        ob.RaceRef(date(2024, 1, 5), 1, 1),
        ob.RaceRef(date(2024, 6, 5), 1, 1),
        ob.RaceRef(date(2024, 6, 6), 1, 1),
    ]
    out = tmp_path / "odds.jsonl"
    ob.backfill(races, out, session=FakeSession(real_page), limiter=no_wait(), workers=2)

    table = ob.coverage(out)
    assert dict(zip(table["month"], table["races"])) == {1: 1, 6: 2}
