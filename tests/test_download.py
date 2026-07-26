"""Network-free tests for the downloader.

Every HTTP interaction goes through an injected fake session, so these tests
prove the URL rules, idempotency and error handling without touching the
origin servers.
"""

from __future__ import annotations

import threading
import time
from datetime import date
from pathlib import Path

import pytest

from kyotei import download as dl
from kyotei.download import Ledger, Status, Summary


class FakeResponse:
    def __init__(self, status_code: int, content: bytes = b"") -> None:
        self.status_code = status_code
        self.content = content


class FakeSession:
    """Serves canned responses and records every URL requested."""

    def __init__(self, responses: dict[str, FakeResponse], default: FakeResponse | None = None):
        self.responses = responses
        self.default = default or FakeResponse(404)
        self.requested: list[str] = []

    def get(self, url: str, timeout: int = 0) -> FakeResponse:
        self.requested.append(url)
        return self.responses.get(url, self.default)


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    """Redirect all filesystem writes into tmp_path."""
    monkeypatch.setattr(dl, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(dl, "EXTRACTED_DIR", tmp_path / "extracted")
    monkeypatch.setattr(dl, "LEDGER_PATH", tmp_path / "raw" / "_ledger.json")

    import kyotei.paths as paths

    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(paths, "EXTRACTED_DIR", tmp_path / "extracted")
    monkeypatch.setattr(paths, "LEDGER_PATH", tmp_path / "raw" / "_ledger.json")
    return tmp_path


def no_sleep() -> None:
    """Replaces the 1s politeness sleep in tests."""


# --------------------------------------------------------------------------
# URL rules (SPEC §3.1, §3.3)
# --------------------------------------------------------------------------


def test_daily_url_matches_spec_example():
    # SPEC §3.1: 2020年7月14日 -> .../od2/B/202007/b200714.lzh
    assert (
        dl.daily_url("B", date(2020, 7, 14))
        == "https://www1.mbrace.or.jp/od2/B/202007/b200714.lzh"
    )
    assert (
        dl.daily_url("K", date(2020, 7, 14))
        == "https://www1.mbrace.or.jp/od2/K/202007/k200714.lzh"
    )


def test_daily_url_rejects_unknown_kind():
    with pytest.raises(ValueError):
        dl.daily_url("X", date(2020, 7, 14))


@pytest.mark.parametrize(
    ("year", "half", "stem"),
    [
        (2015, 1, "fan1410"),  # SPEC §3.3 worked example
        (2015, 2, "fan1504"),  # SPEC §3.3 worked example
        (2020, 1, "fan1910"),
        (2020, 2, "fan2004"),
        (2026, 1, "fan2510"),
        (2026, 2, "fan2604"),
    ],
)
def test_fan_stem_rule(year, half, stem):
    assert dl.fan_stem(year, half) == stem


def test_fan_stem_rejects_bad_half():
    with pytest.raises(ValueError):
        dl.fan_stem(2020, 3)


def test_fan_url_is_on_boatrace_jp():
    assert dl.fan_url(2015, 1).endswith("/kibetsu/fan1410.lzh")
    assert dl.fan_url(2015, 1).startswith("https://www.boatrace.jp/")


def test_fan_file_count_2015_to_2026_is_24():
    stems = [dl.fan_stem(y, h) for y in range(2015, 2027) for h in (1, 2)]
    assert len(stems) == 24
    assert len(set(stems)) == 24, "half-yearly stems must be unique"


# --------------------------------------------------------------------------
# Date iteration
# --------------------------------------------------------------------------


def test_daterange_is_inclusive():
    days = list(dl.daterange(date(2020, 1, 30), date(2020, 2, 2)))
    assert days == [date(2020, 1, 30), date(2020, 1, 31), date(2020, 2, 1), date(2020, 2, 2)]


def test_daterange_rejects_reversed_range():
    with pytest.raises(ValueError):
        list(dl.daterange(date(2020, 2, 2), date(2020, 1, 1)))


# --------------------------------------------------------------------------
# Fetch behaviour
# --------------------------------------------------------------------------


def test_fetch_writes_file_and_leaves_no_partial(tmp_path):
    url = dl.daily_url("B", date(2020, 7, 14))
    session = FakeSession({url: FakeResponse(200, b"PAYLOAD")})
    dest = tmp_path / "raw" / "B" / "202007" / "b200714.lzh"

    status, code = dl.fetch(session, url, dest, sleeper=no_sleep)

    assert (status, code) == (Status.DOWNLOADED, 200)
    assert dest.read_bytes() == b"PAYLOAD"
    assert list(dest.parent.glob("*.part")) == [], "no partial file may survive"


def test_fetch_treats_404_as_missing_without_writing(tmp_path):
    url = dl.daily_url("B", date(2020, 1, 1))
    session = FakeSession({}, default=FakeResponse(404))
    dest = tmp_path / "b.lzh"

    status, code = dl.fetch(session, url, dest, sleeper=no_sleep)

    assert (status, code) == (Status.MISSING, 404)
    assert not dest.exists()


def test_fetch_retries_transient_then_reports_error(tmp_path):
    url = "https://example.invalid/x.lzh"
    session = FakeSession({}, default=FakeResponse(503))

    status, code = dl.fetch(session, url, tmp_path / "x.lzh", sleeper=no_sleep, retries=2)

    assert (status, code) == (Status.ERROR, 503)
    assert len(session.requested) == 3, "1 initial attempt + 2 retries"


def test_fetch_does_not_retry_403(tmp_path):
    """A policy denial must be reported, not retried (see /root/.ccr/README.md)."""
    session = FakeSession({}, default=FakeResponse(403))

    status, code = dl.fetch(session, "https://x/y.lzh", tmp_path / "y.lzh", sleeper=no_sleep)

    assert (status, code) == (Status.ERROR, 403)
    assert len(session.requested) == 1


# --------------------------------------------------------------------------
# Idempotency (SPEC §3.1: re-runs must skip, never re-fetch)
# --------------------------------------------------------------------------


def test_existing_archive_is_skipped_without_any_request():
    day = date(2020, 7, 14)
    dest = dl.raw_path("B", day)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"already here")

    session = FakeSession({})
    summary = dl.download_daily(session, "B", day, day, sleeper=no_sleep)

    assert session.requested == [], "must not re-request an archive already on disk"
    assert summary.counts[Status.SKIPPED.value] == 1
    assert dest.read_bytes() == b"already here", "existing archive must not be touched"


def test_zero_byte_archive_is_refetched():
    day = date(2020, 7, 14)
    dest = dl.raw_path("B", day)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"")

    url = dl.daily_url("B", day)
    session = FakeSession({url: FakeResponse(200, b"REAL")})
    summary = dl.download_daily(session, "B", day, day, sleeper=no_sleep)

    assert summary.counts[Status.DOWNLOADED.value] == 1
    assert dest.read_bytes() == b"REAL"


def test_second_run_does_not_rerequest_known_404_days():
    start, end = date(2020, 1, 1), date(2020, 1, 3)
    session = FakeSession({}, default=FakeResponse(404))

    first = dl.download_daily(session, "B", start, end, sleeper=no_sleep)
    assert first.counts[Status.MISSING.value] == 3
    assert len(session.requested) == 3

    # Fresh Ledger instance -> proves the 404s were persisted to disk.
    session2 = FakeSession({}, default=FakeResponse(404))
    second = dl.download_daily(session2, "B", start, end, sleeper=no_sleep)

    assert session2.requested == [], "known-404 days must not be re-requested"
    assert second.counts[Status.KNOWN_MISSING.value] == 3


def test_ledger_does_not_memoize_errors():
    """A 503 today may be a 200 tomorrow -- only 404s are permanent."""
    day = date(2020, 5, 5)
    session = FakeSession({}, default=FakeResponse(503))
    dl.download_daily(session, "B", day, day, sleeper=no_sleep)

    ledger = Ledger(dl.LEDGER_PATH)
    assert not ledger.is_missing("B", day)


def test_ledger_keys_are_per_kind():
    day = date(2020, 6, 6)
    ledger = Ledger(dl.LEDGER_PATH)
    ledger.record("B", day, Status.MISSING)
    assert ledger.is_missing("B", day)
    assert not ledger.is_missing("K", day), "B and K must not share ledger entries"


# --------------------------------------------------------------------------
# Summary arithmetic -- this is what Phase 1 reports
# --------------------------------------------------------------------------


def test_success_rate_excludes_404_non_race_days():
    summary = Summary()
    summary.add(Status.DOWNLOADED)
    summary.add(Status.DOWNLOADED)
    summary.add(Status.MISSING)  # non-race day: not a failure
    summary.add(Status.MISSING)

    assert summary.attempted == 4
    assert summary.available == 2
    assert summary.success_rate == 1.0


def test_success_rate_counts_errors_as_failures():
    summary = Summary()
    for _ in range(9):
        summary.add(Status.DOWNLOADED)
    summary.add(Status.ERROR, "https://x", 500)

    assert summary.success_rate == pytest.approx(0.9)
    assert summary.errors == [("https://x", 500)]


def test_summary_render_mentions_every_status():
    summary = Summary()
    summary.add(Status.DOWNLOADED)
    text = summary.render()
    for label in ("downloaded", "skipped", "missing", "errors", "success rate"):
        assert label in text


# --------------------------------------------------------------------------
# Extraction: failures must raise, never be swallowed (SPEC §7)
# --------------------------------------------------------------------------


def test_extract_bytes_raises_on_garbage(tmp_path):
    bad = tmp_path / "bad.lzh"
    bad.write_bytes(b"this is definitely not an LZH archive")

    with pytest.raises(dl.ExtractError):
        dl.extract_bytes(bad)


def test_extract_bytes_raises_on_missing_file(tmp_path):
    with pytest.raises(dl.ExtractError):
        dl.extract_bytes(tmp_path / "nope.lzh")


def test_extract_daily_returns_none_for_absent_archive():
    assert dl.extract_daily("B", date(2020, 7, 14)) is None


def test_extract_all_daily_collects_failures_instead_of_crashing():
    day = date(2020, 7, 14)
    dest = dl.raw_path("K", day)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"corrupt")

    extracted, absent, failures = dl.extract_all_daily("K", day, day)

    assert (extracted, absent) == (0, 0)
    assert len(failures) == 1 and failures[0][0] == day


# --------------------------------------------------------------------------
# Politeness
# --------------------------------------------------------------------------


def test_every_request_is_preceded_by_a_sleep():
    calls: list[str] = []
    start, end = date(2020, 1, 1), date(2020, 1, 4)
    session = FakeSession({}, default=FakeResponse(404))

    def counting_sleeper() -> None:
        calls.append("sleep")

    dl.download_daily(session, "B", start, end, sleeper=counting_sleeper)

    assert len(calls) == len(session.requested) == 4


def test_configured_sleep_is_about_one_second():
    assert 0.8 <= dl.SLEEP_SECONDS <= 1.5


# --------------------------------------------------------------------------
# Rate limiting: the politeness budget is on the aggregate request rate, so
# it must hold no matter how many workers run.
# --------------------------------------------------------------------------


class FakeClock:
    """Deterministic clock: sleeping advances time instead of blocking."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_rate_limiter_spaces_calls_by_the_interval():
    clock = FakeClock()
    limiter = dl.RateLimiter(1.0, clock=clock.monotonic, sleep=clock.sleep)

    for _ in range(4):
        limiter.acquire()

    # The first call goes immediately; each later one waits a full interval.
    assert clock.sleeps == [1.0, 1.0, 1.0]
    assert clock.now == pytest.approx(3.0)


def test_rate_limiter_honours_a_faster_rate():
    clock = FakeClock()
    limiter = dl.RateLimiter(4.0, clock=clock.monotonic, sleep=clock.sleep)
    for _ in range(5):
        limiter.acquire()
    assert clock.now == pytest.approx(1.0)  # 4 req/s -> 5th call at t=1.0


def test_rate_limiter_does_not_wait_when_callers_are_already_late():
    clock = FakeClock()
    limiter = dl.RateLimiter(1.0, clock=clock.monotonic, sleep=clock.sleep)
    limiter.acquire()
    clock.now = 100.0  # a slow response already consumed the budget
    limiter.acquire()
    assert clock.now == 100.0, "no artificial delay once the schedule is behind"


def test_rate_limiter_rejects_non_positive_rate():
    with pytest.raises(ValueError):
        dl.RateLimiter(0)


def test_rate_limiter_is_thread_safe_and_never_exceeds_the_budget():
    limiter = dl.RateLimiter(1000.0)  # fast, but still serialised
    stamps: list[float] = []
    lock = threading.Lock()

    def worker() -> None:
        for _ in range(5):
            limiter.acquire()
            with lock:
                stamps.append(time.monotonic())

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(stamps) == 40, "every acquire must be accounted for"


# --------------------------------------------------------------------------
# Concurrent download keeps the sequential guarantees
# --------------------------------------------------------------------------


class LockedFakeSession(FakeSession):
    """Thread-safe fake so concurrent tests do not race on `requested`."""

    def __init__(self, responses, default=None):
        super().__init__(responses, default)
        import threading

        self._lock = threading.Lock()

    def get(self, url: str, timeout: int = 0) -> FakeResponse:
        with self._lock:
            self.requested.append(url)
        return self.responses.get(url, self.default)


def _no_wait_limiter():
    clock = FakeClock()
    return dl.RateLimiter(1.0, clock=clock.monotonic, sleep=lambda s: None)


def test_concurrent_download_fetches_every_day_exactly_once():
    start, end = date(2020, 3, 1), date(2020, 3, 10)
    urls = {dl.daily_url("B", d): FakeResponse(200, b"X") for d in dl.daterange(start, end)}
    session = LockedFakeSession(urls)

    summary = dl.download_daily_concurrent(
        session, "B", start, end, limiter=_no_wait_limiter(), workers=4
    )

    assert summary.counts[Status.DOWNLOADED.value] == 10
    assert sorted(session.requested) == sorted(urls), "each day requested once"


def test_concurrent_download_skips_files_already_on_disk():
    start, end = date(2020, 3, 1), date(2020, 3, 3)
    held = dl.raw_path("B", date(2020, 3, 2))
    held.parent.mkdir(parents=True, exist_ok=True)
    held.write_bytes(b"already")

    urls = {
        dl.daily_url("B", d): FakeResponse(200, b"X")
        for d in (date(2020, 3, 1), date(2020, 3, 3))
    }
    session = LockedFakeSession(urls)

    summary = dl.download_daily_concurrent(
        session, "B", start, end, limiter=_no_wait_limiter(), workers=4
    )

    assert summary.counts[Status.SKIPPED.value] == 1
    assert summary.counts[Status.DOWNLOADED.value] == 2
    assert dl.daily_url("B", date(2020, 3, 2)) not in session.requested


def test_concurrent_download_persists_404s_to_the_ledger():
    start, end = date(2020, 4, 1), date(2020, 4, 5)
    session = LockedFakeSession({}, default=FakeResponse(404))

    first = dl.download_daily_concurrent(
        session, "B", start, end, limiter=_no_wait_limiter(), workers=4
    )
    assert first.counts[Status.MISSING.value] == 5

    session2 = LockedFakeSession({}, default=FakeResponse(404))
    second = dl.download_daily_concurrent(
        session2, "B", start, end, limiter=_no_wait_limiter(), workers=4
    )
    assert session2.requested == []
    assert second.counts[Status.KNOWN_MISSING.value] == 5


def test_concurrent_download_reports_errors_without_losing_other_days():
    start, end = date(2020, 5, 1), date(2020, 5, 4)
    urls = {dl.daily_url("B", d): FakeResponse(200, b"X") for d in dl.daterange(start, end)}
    urls[dl.daily_url("B", date(2020, 5, 3))] = FakeResponse(500)
    session = LockedFakeSession(urls)

    summary = dl.download_daily_concurrent(
        session, "B", start, end, limiter=_no_wait_limiter(), workers=4
    )

    assert summary.counts[Status.DOWNLOADED.value] == 3
    assert summary.counts[Status.ERROR.value] == 1
    assert summary.errors[0][1] == 500


def test_session_pool_gives_each_thread_its_own_session():
    class OneSession:
        def get(self, url: str, timeout: int = 0) -> FakeResponse:
            return FakeResponse(200, b"X")

    pool = dl.SessionPool(factory=OneSession)
    # Hold the objects, not their ids: CPython recycles addresses once a
    # short-lived session is collected, which would fake a collision.
    seen: list[object] = []
    lock = threading.Lock()

    def worker():
        pool.get("https://x")
        with lock:
            seen.append(pool._local.session)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(seen) == 4
    assert len({id(s) for s in seen}) == 4, "each thread must hold a distinct session"
