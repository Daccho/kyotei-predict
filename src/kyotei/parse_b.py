"""Parser for the 番組表 (race card) daily feed.

Provenance of the offsets
-------------------------
SPEC §3.2 points at https://www.boatrace.jp/owpc/pc/extra/data/layout.html as
the layout reference. That page turned out to document only the
「モーターボートファン手帳」(the fan{YYMM} half-yearly file); it contains no B or
K layout, and the mbrace download pages carry no layout document either. The
B/K feeds are instead self-describing: every race block repeats its own column
header, and the entry rows are a fixed 79-byte record.

The offsets below were therefore transcribed from those in-file column headers
and then cross-validated against independent data rather than eyeballed:
  * motor/boat numbers parsed here must equal the motor/boat numbers the K feed
    reports for the same racer in the same race (see tests/test_parse_pilot.py);
  * every race must yield exactly 6 rows;
  * per-year record success rates are reported so a layout change in any year
    shows up as a cliff (SPEC §3.2).

File structure
--------------
    STARTB                     file start
    24BBGN                     stadium 24 section begins
      ... header lines ...
      　１Ｒ  一般（Ｂ組）          Ｈ１８００ｍ  電話投票締切予定１５：１５
      -----
      艇 選手 …                 repeated column header
      -----
      1 4164岩永節也37長崎51B1 …  6 entry rows, 79 bytes each
    24BEND                     stadium section ends
    FINALB
A single daily file carries every stadium racing that day.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, time
from typing import Any

from kyotei.fixedwidth import Field, Layout, iter_lines
from kyotei.parse_stats import FileOutcome

#: One entry row. Byte offsets, not character offsets: the name and branch
#: fields are 全角 and would shift every later column if decoded first.
ENTRY_LAYOUT = Layout(
    "b_entry",
    [
        Field("lane", 0, 1, "int", allow_blank=False),
        Field("racer_id", 2, 4, "int"),
        Field("racer_name", 6, 8, "text"),          # 4 全角 chars, names are truncated
        Field("age", 14, 2, "int"),
        Field("branch", 16, 4, "text"),             # 2 全角 chars
        Field("weight_kg", 20, 2, "float"),
        Field("grade", 22, 2, "text"),
        Field("national_win_rate", 24, 5, "float"),
        Field("national_top2_rate", 29, 6, "float"),
        Field("local_win_rate", 35, 5, "float"),
        Field("local_top2_rate", 40, 6, "float"),
        Field("motor_no", 46, 3, "int"),
        Field("motor_top2_rate", 49, 6, "float"),
        Field("boat_no", 55, 3, "int"),
        Field("boat_top2_rate", 58, 6, "float"),
        Field("series_results", 64, 12, "text"),    # 今節成績: 6 slots, refined in Phase 2
    ],
)

ENTRY_RECORD_BYTES = 79

STADIUM_BEGIN = re.compile(rb"^(\d{2})BBGN\s*$")
STADIUM_END = re.compile(rb"^(\d{2})BEND\s*$")
ENTRY_ROW = re.compile(rb"^[1-6] ?\d{4}")

#: Race header, e.g. '　１Ｒ  一般（Ｂ組）          Ｈ１８００ｍ  電話投票締切予定１５：１５'
RACE_HEADER = re.compile(
    r"^\s*(?P<race_no>\d{1,2})R\s+(?P<rest>.*?)"
    r"H(?P<distance>\d{3,4})m\s*"
    r"(?:電話投票締切予定\s*(?P<hh>\d{1,2}):(?P<mm>\d{2}))?"
)
SERIES_DAY = re.compile(r"第\s*(\d+)\s*日")

_ZEN_TO_HAN = str.maketrans(
    "０１２３４５６７８９"
    "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
    # Lower case matters: the distance is published as 'Ｈ１８００ｍ' with a
    # full-width lower-case ｍ, so omitting these makes every header unmatchable.
    "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ"
    "：－．",
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    ":-.",
)

TITLE_SLICE = slice(8, 20)     # 6 全角 chars
FLAGS_SLICE = slice(20, 30)    # holds 進入固定 when the race fixes courses


def normalise(text: str) -> str:
    """Full-width digits/letters/colon to ASCII, 全角 space to ASCII space."""
    return text.translate(_ZEN_TO_HAN).replace("　", " ")


@dataclass
class RaceCard:
    race_date: date
    stadium_id: int
    race_no: int
    title: str | None = None
    distance_m: int | None = None
    deadline_time: time | None = None
    fixed_course: bool = False
    series_day: int | None = None
    entries: list[dict[str, Any]] = field(default_factory=list)

    @property
    def key(self) -> tuple[date, int, int]:
        return (self.race_date, self.stadium_id, self.race_no)


def parse_race_header(line: bytes) -> dict[str, Any] | None:
    """Interpret a race header row, or return None when it is not one."""
    try:
        raw = line.decode("cp932")
    except UnicodeDecodeError:
        return None
    if "電話投票締切" not in raw and "H1" not in normalise(raw):
        return None
    flat = normalise(raw)
    m = RACE_HEADER.match(flat)
    if not m:
        return None

    deadline = None
    if m.group("hh") is not None:
        hh, mm = int(m.group("hh")), int(m.group("mm"))
        if hh < 24 and mm < 60:
            deadline = time(hh, mm)

    title = line[TITLE_SLICE].decode("cp932", errors="replace").replace("　", "").strip()
    flags = line[FLAGS_SLICE].decode("cp932", errors="replace")

    return {
        "race_no": int(m.group("race_no")),
        "title": title or None,
        "distance_m": int(m.group("distance")),
        "deadline_time": deadline,
        "fixed_course": "進入固定" in flags,
    }


def parse_payload(payload: bytes, day: date, source: str) -> tuple[list[RaceCard], FileOutcome]:
    """Parse a decompressed B file into race cards.

    Errors are counted and sampled, never swallowed (SPEC §7).
    """
    outcome = FileOutcome(source=source, day=day)
    races: list[RaceCard] = []
    stadium: int | None = None
    current: RaceCard | None = None
    series_day: int | None = None

    def note(msg: str) -> None:
        outcome.records_failed += 1
        if len(outcome.samples) < 5:
            outcome.samples.append(msg)

    for line in iter_lines(payload):
        begin = STADIUM_BEGIN.match(line)
        if begin:
            stadium = int(begin.group(1))
            current = None
            series_day = None
            continue
        if STADIUM_END.match(line):
            stadium = current = None
            continue
        if stadium is None:
            continue

        if series_day is None and "第".encode("cp932") in line:
            found = SERIES_DAY.search(normalise(line.decode("cp932", errors="replace")))
            if found:
                series_day = int(found.group(1))

        header = parse_race_header(line)
        if header:
            current = RaceCard(
                race_date=day,
                stadium_id=stadium,
                series_day=series_day,
                **header,
            )
            races.append(current)
            continue

        if not ENTRY_ROW.match(line) or len(line) < ENTRY_RECORD_BYTES:
            continue
        if current is None:
            note(f"entry row before any race header: {line[:20]!r}")
            continue

        parsed, errors = ENTRY_LAYOUT.parse(line)
        if errors:
            note(f"R{current.race_no} lane {parsed.get('lane')}: {errors[0]}")
            # A single bad column is kept as NULL rather than dropping the boat;
            # dropping would silently break the 6-rows-per-race invariant.
        current.entries.append(parsed)
        outcome.records_ok += 1

    return races, outcome
