"""Integration tests against a real Postgres carrying sql/schema.sql.

These prove the guardrails actually fire, rather than merely being written down:
  * a race must have exactly 6 entries (v_bad_entry_counts)
  * two boats cannot share an actual course
  * lane and course are independent columns
  * dividends are keyed so a re-ingest updates instead of duplicating

Skipped when no database is reachable (e.g. a fresh container before
scripts/setup_db.sh has run).
"""

from __future__ import annotations

import os
from datetime import date

import pytest

psycopg = pytest.importorskip("psycopg")

DSN = os.environ.get("KYOTEI_DSN", "postgresql://kyotei:kyotei@localhost:5432/kyotei")
RACE_DAY = date(2020, 7, 14)
STADIUM = 4
RACE_NO = 3


@pytest.fixture()
def conn():
    """A connection whose work is always rolled back."""
    try:
        connection = psycopg.connect(DSN, connect_timeout=5)
    except Exception as exc:  # noqa: BLE001 - environment probe, reported not hidden
        pytest.skip(f"no database at {DSN}: {exc}")
    connection.autocommit = False
    try:
        yield connection
    finally:
        connection.rollback()
        connection.close()


def insert_race(cur, *, has_k: bool = True) -> None:
    cur.execute(
        """
        INSERT INTO races (race_date, stadium_id, race_no, distance_m, has_b, has_k)
        VALUES (%s, %s, %s, 1800, TRUE, %s)
        """,
        (RACE_DAY, STADIUM, RACE_NO, has_k),
    )


def insert_entry(cur, lane: int, course: int | None, finish: int | None = None) -> None:
    cur.execute(
        """
        INSERT INTO entries (race_date, stadium_id, race_no, lane, course,
                             racer_id, grade, finish_position)
        VALUES (%s, %s, %s, %s, %s, %s, 'A1', %s)
        """,
        (RACE_DAY, STADIUM, RACE_NO, lane, course, 4000 + lane, finish),
    )


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------


def test_all_24_stadiums_are_seeded(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT count(*), min(stadium_id), max(stadium_id) FROM stadiums")
        count, lo, hi = cur.fetchone()
    assert (count, lo, hi) == (24, 1, 24)


# ---------------------------------------------------------------------------
# lane vs course: the distinction SPEC §2/Phase 1 requires
# ---------------------------------------------------------------------------


def test_lane_and_course_are_independent_columns(conn):
    """A boat drawn in lane 4 may start from course 2 (前づけ)."""
    with conn.cursor() as cur:
        insert_race(cur)
        insert_entry(cur, lane=4, course=2)
        cur.execute(
            """
            SELECT lane, course FROM entries
            WHERE race_date = %s AND stadium_id = %s AND race_no = %s
            """,
            (RACE_DAY, STADIUM, RACE_NO),
        )
        assert cur.fetchone() == (4, 2)


def test_two_boats_cannot_share_a_course(conn):
    with conn.cursor() as cur:
        insert_race(cur)
        insert_entry(cur, lane=1, course=1)
        with pytest.raises(psycopg.errors.UniqueViolation):
            insert_entry(cur, lane=2, course=1)


def test_course_may_be_null_before_results_land(conn):
    """Several boats can have NULL course: the partial index must not collide."""
    with conn.cursor() as cur:
        insert_race(cur, has_k=False)
        for lane in range(1, 7):
            insert_entry(cur, lane=lane, course=None)
        cur.execute(
            "SELECT count(*) FROM entries WHERE race_date = %s AND course IS NULL",
            (RACE_DAY,),
        )
        assert cur.fetchone()[0] == 6


def test_course_out_of_range_is_rejected(conn):
    with conn.cursor() as cur:
        insert_race(cur)
        with pytest.raises(psycopg.errors.CheckViolation):
            insert_entry(cur, lane=1, course=7)


def test_duplicate_lane_in_one_race_is_rejected(conn):
    with conn.cursor() as cur:
        insert_race(cur)
        insert_entry(cur, lane=1, course=1)
        with pytest.raises(psycopg.errors.UniqueViolation):
            insert_entry(cur, lane=1, course=2)


# ---------------------------------------------------------------------------
# "1 race = 6 rows" detection
# ---------------------------------------------------------------------------


def test_bad_entry_count_view_flags_a_five_boat_race(conn):
    with conn.cursor() as cur:
        insert_race(cur)
        for lane in range(1, 6):  # only 5 boats
            insert_entry(cur, lane=lane, course=lane)
        cur.execute(
            """
            SELECT entry_count FROM v_bad_entry_counts
            WHERE race_date = %s AND stadium_id = %s AND race_no = %s
            """,
            (RACE_DAY, STADIUM, RACE_NO),
        )
        assert cur.fetchone() == (5,)


def test_bad_entry_count_view_passes_a_six_boat_race(conn):
    with conn.cursor() as cur:
        insert_race(cur)
        for lane in range(1, 7):
            insert_entry(cur, lane=lane, course=lane)
        cur.execute(
            """
            SELECT count(*) FROM v_bad_entry_counts
            WHERE race_date = %s AND stadium_id = %s AND race_no = %s
            """,
            (RACE_DAY, STADIUM, RACE_NO),
        )
        assert cur.fetchone()[0] == 0


def test_race_with_zero_entries_is_flagged(conn):
    """A B file that parsed a header but no boats must not pass silently."""
    with conn.cursor() as cur:
        insert_race(cur)
        cur.execute(
            "SELECT entry_count FROM v_bad_entry_counts WHERE race_date = %s",
            (RACE_DAY,),
        )
        assert cur.fetchone() == (0,)


# ---------------------------------------------------------------------------
# Referential integrity and idempotent re-ingest
# ---------------------------------------------------------------------------


def test_entry_without_race_is_rejected(conn):
    with conn.cursor() as cur:
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            insert_entry(cur, lane=1, course=1)


def test_reingesting_a_payout_updates_instead_of_duplicating(conn):
    with conn.cursor() as cur:
        insert_race(cur)
        for _ in range(2):
            cur.execute(
                """
                INSERT INTO payouts (race_date, stadium_id, race_no, bet_type,
                                     combination, payout_yen, popularity)
                VALUES (%s, %s, %s, 'trifecta', '1-2-3', 1230, 4)
                ON CONFLICT (race_date, stadium_id, race_no, bet_type, combination)
                DO UPDATE SET payout_yen = EXCLUDED.payout_yen,
                              popularity = EXCLUDED.popularity
                """,
                (RACE_DAY, STADIUM, RACE_NO),
            )
        cur.execute(
            "SELECT count(*), max(payout_yen) FROM payouts WHERE race_date = %s",
            (RACE_DAY,),
        )
        assert cur.fetchone() == (1, 1230)


def test_unknown_bet_type_is_rejected(conn):
    with conn.cursor() as cur:
        insert_race(cur)
        with pytest.raises(psycopg.errors.CheckViolation):
            cur.execute(
                """
                INSERT INTO payouts (race_date, stadium_id, race_no, bet_type,
                                     combination, payout_yen)
                VALUES (%s, %s, %s, 'sanrentan', '1-2-3', 1230)
                """,
                (RACE_DAY, STADIUM, RACE_NO),
            )


def test_missing_trifecta_view_flags_a_finished_race_without_dividend(conn):
    with conn.cursor() as cur:
        insert_race(cur, has_k=True)
        cur.execute(
            """
            SELECT count(*) FROM v_missing_trifecta
            WHERE race_date = %s AND stadium_id = %s AND race_no = %s
            """,
            (RACE_DAY, STADIUM, RACE_NO),
        )
        assert cur.fetchone()[0] == 1


# ---------------------------------------------------------------------------
# The leak-relevant column on racers
# ---------------------------------------------------------------------------


def test_racers_period_carries_an_effective_from_date(conn):
    """Period figures describe the past, so features join as-of publication."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO racers (racer_id, period_year, period_half, effective_from,
                                win_rate, starts)
            VALUES (4321, 2020, 1, %s, 6.12, 120)
            RETURNING effective_from
            """,
            (date(2020, 5, 1),),
        )
        assert cur.fetchone()[0] == date(2020, 5, 1)
