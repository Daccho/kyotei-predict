"""Race grade (SG / G1 / G2 / G3 / 一般 …) from the monthly schedule.

Why this is a separate source
-----------------------------
The B/K daily feeds cover every race at all 24 stadiums, so coverage was never
the gap. What they do not carry is the *grade* of the series a race belongs to.
A race header says '予選' or '優勝戦'; it does not say whether that heat is part
of an SG worth tens of millions or an ordinary weekday meeting. The two have
very different fields, and the course-1 advantage differs between them, so a
model that cannot tell them apart is averaging over two different sports.

The monthly schedule page carries it in machine-readable form:

    /owpc/pc/race/monthlyschedule?ym=YYYYMM

Each row is a stadium (identified by ``jcd`` in the row header's link), each
column a day, and each series cell carries a ``is-gradeColorXxx`` class plus a
``colspan`` equal to the number of days the series runs.

Cost is negligible next to the odds backfill: one page per month, so about 139
pages for 2015-2026 -- a couple of minutes at 1 req/s.
"""

from __future__ import annotations

import argparse
import calendar
import html as html_module
import re
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import polars as pl

MONTHLY_URL = "https://www.boatrace.jp/owpc/pc/race/monthlyschedule"

#: Class suffix -> the label used downstream. Ordered strongest first so the
#: encoding in features.py is monotone in prestige, which is the thing that
#: actually correlates with field strength.
GRADES: dict[str, str] = {
    "SG": "SG",
    "G1": "G1",
    "G2": "G2",
    "G3": "G3",
    "Takumi": "マスターズ",
    "Venus": "ヴィーナス",
    "Lady": "オールレディース",
    "Rookie": "ルーキー",
    "Ippan": "一般",
}
#: Rank used as the numeric feature. Lower is a bigger meeting.
GRADE_RANK: dict[str, int] = {name: i for i, name in enumerate(GRADES.values())}


def monthly_url(year: int, month: int) -> str:
    return f"{MONTHLY_URL}?ym={year:04d}{month:02d}"


@dataclass(frozen=True)
class SeriesDay:
    race_date: date
    stadium_id: int
    grade: str
    series: str


def _text(markup: str) -> str:
    return html_module.unescape(re.sub(r"<[^>]+>", "", markup)).strip()


def _header_days(header_row: str, year: int, month: int) -> list[date | None]:
    """Map the header's day-of-month labels to real dates.

    The grid starts a few days before the first of the month, so a leading run
    of large day numbers belongs to the previous month. The switch happens when
    the numbers reset.
    """
    cells = re.findall(r"(?s)<t[dh][^>]*>(.*?)</t[dh]>", header_row)
    days: list[date | None] = []
    current_year, current_month = year, month
    started = False
    for cell in cells[1:]:  # first cell is the 'ボートレース場' label
        match = re.match(r"(\d{1,2})", _text(cell))
        if not match:
            days.append(None)
            continue
        number = int(match.group(1))
        if not started and number > 20:
            # Still in the tail of the previous month.
            previous = date(year, month, 1) - timedelta(days=1)
            current_year, current_month = previous.year, previous.month
        else:
            started = True
            current_year, current_month = year, month
        last = calendar.monthrange(current_year, current_month)[1]
        days.append(date(current_year, current_month, number) if number <= last else None)
    return days


def parse_monthly_schedule(markup: str, year: int, month: int) -> list[SeriesDay]:
    """Every (stadium, day) that has a series, with its grade."""
    body = re.sub(r"(?s)<(script|style)\b.*?</\1>", " ", markup)
    out: list[SeriesDay] = []

    for table in re.findall(r"(?s)<table[^>]*>(.*?)</table>", body):
        rows = re.findall(r"(?s)<tr[^>]*>(.*?)</tr>", table)
        if not rows:
            continue
        days = _header_days(rows[0], year, month)
        if not any(days):
            continue

        for row in rows[1:]:
            stadium = re.search(r"jcd=(\d{1,2})", row)
            if not stadium:
                continue
            stadium_id = int(stadium.group(1))

            column = 0
            for attrs, cell in re.findall(r"(?s)<td([^>]*)>(.*?)</td>", row):
                if "jcd=" in cell and "gradeColor" not in attrs:
                    continue  # the row header cell itself
                span_match = re.search(r'colspan\s*=\s*"?(\d+)"?', attrs)
                span = int(span_match.group(1)) if span_match else 1
                grade_match = re.search(r"is-gradeColor(\w+)", attrs)
                if grade_match:
                    grade = GRADES.get(grade_match.group(1), grade_match.group(1))
                    series = _text(cell)
                    for offset in range(span):
                        index = column + offset
                        if index < len(days) and days[index] is not None:
                            out.append(SeriesDay(days[index], stadium_id, grade, series))
                column += span
    return out


def to_frame(entries: list[SeriesDay]) -> pl.DataFrame:
    schema = {
        "race_date": pl.Date,
        "stadium_id": pl.Int16,
        "grade": pl.Utf8,
        "series": pl.Utf8,
    }
    if not entries:
        return pl.DataFrame(schema=schema)
    return (
        pl.DataFrame([e.__dict__ for e in entries], schema=schema)
        # A stadium runs one series a day; duplicates would fan out the join
        # onto races, so they are collapsed here rather than downstream.
        .unique(subset=["race_date", "stadium_id"], keep="first")
        .sort(["race_date", "stadium_id"])
    )


def attach_grade(races: pl.DataFrame, schedule: pl.DataFrame) -> pl.DataFrame:
    """Join the grade onto races, asserting the row count is unchanged."""
    before = races.height
    joined = races.join(
        schedule.select(["race_date", "stadium_id", "grade", "series"]),
        on=["race_date", "stadium_id"],
        how="left",
    )
    if joined.height != before:
        raise ValueError(f"grade join changed the row count: {before} -> {joined.height}")
    return joined


def months(start: date, end: date) -> list[tuple[int, int]]:
    out, year, month = [], start.year, start.month
    while (year, month) <= (end.year, end.month):
        out.append((year, month))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


def main(argv: list[str] | None = None) -> int:
    from kyotei.download import RateLimiter, SessionPool
    from kyotei.paths import PARQUET_DIR
    from kyotei.scrape import fetch_page

    parser = argparse.ArgumentParser(description="Fetch race grades from the monthly schedule")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2015, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date.today())
    parser.add_argument("--out", default=str(PARQUET_DIR / "schedule.parquet"))
    parser.add_argument("--rate", type=float, default=1.0)
    args = parser.parse_args(argv)

    session, limiter = SessionPool(), RateLimiter(args.rate)
    wanted = months(args.start, args.end)
    print(f"fetching {len(wanted)} monthly pages ({wanted[0]} .. {wanted[-1]})", flush=True)

    collected: list[SeriesDay] = []
    for index, (year, month) in enumerate(wanted, start=1):
        try:
            page = fetch_page(session, monthly_url(year, month), sleeper=limiter.acquire)
        except Exception as exc:  # noqa: BLE001 - reported, not hidden
            print(f"  {year}-{month:02d}: {type(exc).__name__}: {exc}")
            continue
        found = parse_monthly_schedule(page, year, month)
        collected.extend(found)
        if index % 12 == 0 or index == len(wanted):
            print(f"  {year}-{month:02d}: {len(found)} stadium-days "
                  f"(total {len(collected)})", flush=True)

    frame = to_frame(collected)
    print(f"\nstadium-days: {frame.height}")
    if frame.is_empty():
        return 1
    print(frame.group_by("grade").agg(pl.len().alias("days")).sort("days", descending=True))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(out)
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
