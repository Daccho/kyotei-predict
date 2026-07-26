"""Parser for the 競走成績 (results) daily feed.

Offsets come from the file's own repeated column header (see parse_b.py for why
the published layout page does not cover B/K) and are cross-validated: the
motor/boat numbers here must match the B feed for the same racer and race, and
the trifecta dividend's combination must equal the lanes of the first three
finishers. Both checks live in tests/test_parse_pilot.py.

Per-race block
--------------
       1R       一般（Ｂ組）                 H1800m  晴　  風  北　　 2m  波　  1cm
      着 艇 登番 　選　手　名　　ﾓｰﾀｰ ﾎﾞｰﾄ 展示 進入 ｽﾀｰﾄﾀｲﾐﾝｸ ﾚｰｽﾀｲﾑ 差し
     -----
       01  4 4948 木　場　　悠　介 46   36  6.75   4    0.14     1.48.7
       ... 6 rows ...
             単勝     4          490
             ３連単   4-1-2     1730  人気     5

Two columns matter more than any other here: 艇 is the draw (lane) and 進入 is
the course actually taken. They are kept separate all the way to the database.
The 決まり手 (winning move) is not a column -- it is appended to the column
header line, so it is read from there.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from kyotei.fixedwidth import clean_text, iter_lines
from kyotei.parse_stats import FileOutcome

STADIUM_BEGIN = re.compile(rb"^(\d{2})KBGN\s*$")
STADIUM_END = re.compile(rb"^(\d{2})KEND\s*$")

#: Result row: two-space indent, 2-byte 着, 2 spaces, lane, space, 4-digit 登番.
RESULT_ROW = re.compile(rb"^  (?:\d\d|[A-Z0-9 ]{2})  [1-6] \d{4} ")
RESULT_RECORD_BYTES = 60

# Byte slices within a result row.
S_FINISH = slice(2, 4)
S_LANE = slice(6, 7)
S_RACER_ID = slice(8, 12)
S_NAME = slice(13, 29)      # 8 全角 chars, spaced out by the feed
S_MOTOR = slice(29, 33)
S_BOAT = slice(33, 37)
S_EXHIBITION = slice(37, 43)
S_COURSE = slice(43, 47)
S_START = slice(47, 55)
S_RACE_TIME = slice(55, 66)

_ZEN_TO_HAN = str.maketrans("０１２３４５６７８９：－", "0123456789:-")

K_RACE_HEADER = re.compile(
    r"^\s*(?P<race_no>\d{1,2})R\s+(?P<title>.*?)\s+H(?P<distance>\d{3,4})m"
    r"(?:\s+(?P<weather>\S+))?"
    r"(?:\s+風\s+(?:(?P<wind_dir>[^\s\d]+)\s+)?(?P<wind_speed>-?\d+)m)?"
    r"(?:\s+波\s+(?P<wave>-?\d+)cm)?"
)

BET_LABELS = {
    "単勝": "win",
    "複勝": "place",
    "２連単": "exacta",
    "２連複": "quinella",
    "拡連複": "wide",
    "３連単": "trifecta",
    "３連複": "trio",
}
#: Dividends whose combination has no order (sorted so lookups are deterministic).
UNORDERED = {"trio", "quinella", "wide", "place", "win"}

PAYOUT_ENTRY = re.compile(r"(\d(?:-\d){0,2})\s+(\d+)(?:\s+人気\s+(\d+))?")
DECISIONS = ("逃げ", "差し", "まくり差し", "まくり", "抜き", "恵まれ")


def normalise(text: str) -> str:
    return text.translate(_ZEN_TO_HAN).replace("　", " ")


def canonical_combination(bet_type: str, raw: str) -> str:
    """Deterministic combination string.

    Ordered bets keep the published order; unordered bets are sorted so that
    a predicted trio and the published trio always agree.
    """
    parts = raw.split("-")
    if bet_type in UNORDERED:
        parts = sorted(parts)
    return "-".join(parts)


def parse_race_time(text: str) -> float | None:
    """'1.48.7' -> 108.7 seconds. Blank or '.  . ' -> None."""
    cleaned = normalise(text).strip()
    m = re.match(r"^(\d+)\.(\d{1,2})\.(\d)$", cleaned)
    if not m:
        return None
    minutes, seconds, tenths = m.groups()
    return int(minutes) * 60 + int(seconds) + int(tenths) / 10


def parse_start_timing(text: str) -> tuple[float | None, str | None]:
    """Return (timing, flag).

    A flying start is published as 'F.01' and means the boat crossed 0.01s
    early, i.e. a negative timing. Late starts appear as 'L'.
    """
    cleaned = normalise(text).strip()
    if not cleaned:
        return None, None
    if cleaned.startswith("F"):
        rest = cleaned[1:].lstrip(".")
        try:
            return -float(f"0.{rest}") if rest else None, "F"
        except ValueError:
            return None, "F"
    if cleaned.startswith("L"):
        return None, "L"
    try:
        return float(cleaned), None
    except ValueError:
        return None, None


def parse_finish(text: str) -> tuple[int | None, str | None]:
    """'01'..'06' -> position. 'F', 'K0', 'S1', ... -> status only."""
    cleaned = text.strip()
    if cleaned.isdigit():
        value = int(cleaned)
        return (value, None) if 1 <= value <= 6 else (None, cleaned)
    return None, cleaned or None


@dataclass
class RaceResult:
    race_date: date
    stadium_id: int
    race_no: int
    title: str | None = None
    distance_m: int | None = None
    weather: str | None = None
    wind_direction: str | None = None
    wind_speed_m: int | None = None
    wave_height_cm: int | None = None
    decision: str | None = None
    entries: list[dict[str, Any]] = field(default_factory=list)
    payouts: list[dict[str, Any]] = field(default_factory=list)

    @property
    def key(self) -> tuple[date, int, int]:
        return (self.race_date, self.stadium_id, self.race_no)


def parse_k_race_header(line: bytes) -> dict[str, Any] | None:
    try:
        raw = line.decode("cp932")
    except UnicodeDecodeError:
        return None
    flat = normalise(raw)
    if not re.match(r"^\s*\d{1,2}R\s", flat):
        return None
    m = K_RACE_HEADER.match(flat)
    if not m:
        return None
    wind_dir = (m.group("wind_dir") or "").strip() or None
    return {
        "race_no": int(m.group("race_no")),
        "title": (m.group("title") or "").replace(" ", "") or None,
        "distance_m": int(m.group("distance")),
        "weather": (m.group("weather") or "").strip() or None,
        "wind_direction": wind_dir,
        "wind_speed_m": int(m.group("wind_speed")) if m.group("wind_speed") else None,
        "wave_height_cm": int(m.group("wave")) if m.group("wave") else None,
    }


def parse_decision(line: bytes) -> str | None:
    """Read 決まり手 off the column-header line."""
    try:
        raw = line.decode("cp932")
    except UnicodeDecodeError:
        return None
    tail = raw.split("ﾚｰｽﾀｲﾑ")[-1].replace("　", "").strip()
    for name in DECISIONS:  # 'まくり差し' before 'まくり': longest match first
        if tail.startswith(name):
            return name
    return tail or None


def parse_result_row(line: bytes) -> dict[str, Any]:
    finish_position, finish_status = parse_finish(clean_text(line[S_FINISH]))
    timing, flag = parse_start_timing(clean_text(line[S_START]))
    return {
        "finish_position": finish_position,
        "finish_status": finish_status or flag,
        "lane": int(clean_text(line[S_LANE])),
        "course": _int_or_none(clean_text(line[S_COURSE])),
        "racer_id": _int_or_none(clean_text(line[S_RACER_ID])),
        "racer_name": clean_text(line[S_NAME]).replace("　", "").replace(" ", ""),
        "motor_no": _int_or_none(clean_text(line[S_MOTOR])),
        "boat_no": _int_or_none(clean_text(line[S_BOAT])),
        "exhibition_time": _float_or_none(clean_text(line[S_EXHIBITION])),
        "start_timing": timing,
        "race_time_sec": parse_race_time(clean_text(line[S_RACE_TIME])),
    }


def _int_or_none(text: str) -> int | None:
    cleaned = normalise(text).strip()
    return int(cleaned) if cleaned.isdigit() else None


def _float_or_none(text: str) -> float | None:
    cleaned = normalise(text).strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_payout_line(line: str, previous: str | None) -> tuple[list[dict[str, Any]], str | None]:
    """Parse one dividend line; ``previous`` carries the 拡連複 continuation."""
    stripped = line.strip()
    bet_type = None
    for label, name in BET_LABELS.items():
        if stripped.startswith(label):
            bet_type = name
            stripped = stripped[len(label) :]
            break
    if bet_type is None:
        # Unlabelled continuation rows belong to the previous bet type (拡連複
        # publishes three rows). Anything else is not a dividend line.
        if previous is None or not re.match(r"^\d(-\d)*\s+\d+", stripped):
            return [], previous
        bet_type = previous

    out = []
    for combo, yen, pop in PAYOUT_ENTRY.findall(normalise(stripped)):
        out.append(
            {
                "bet_type": bet_type,
                "combination": canonical_combination(bet_type, combo),
                "payout_yen": int(yen),
                "popularity": int(pop) if pop else None,
            }
        )
    return out, bet_type


def parse_payload(payload: bytes, day: date, source: str) -> tuple[list[RaceResult], FileOutcome]:
    outcome = FileOutcome(source=source, day=day)
    races: list[RaceResult] = []
    stadium: int | None = None
    current: RaceResult | None = None
    last_bet: str | None = None
    in_payouts = False

    def note(msg: str) -> None:
        outcome.records_failed += 1
        if len(outcome.samples) < 5:
            outcome.samples.append(msg)

    for line in iter_lines(payload):
        begin = STADIUM_BEGIN.match(line)
        if begin:
            stadium, current, last_bet = int(begin.group(1)), None, None
            continue
        if STADIUM_END.match(line):
            stadium, current, last_bet = None, None, None
            continue
        if stadium is None:
            continue

        header = parse_k_race_header(line)
        if header:
            current = RaceResult(race_date=day, stadium_id=stadium, **header)
            races.append(current)
            last_bet, in_payouts = None, False
            continue

        if current is None:
            continue

        if "着 艇 登番".encode("cp932") in line:
            current.decision = parse_decision(line)
            continue

        if RESULT_ROW.match(line) and len(line) >= RESULT_RECORD_BYTES:
            try:
                current.entries.append(parse_result_row(line))
                outcome.records_ok += 1
            except (ValueError, UnicodeDecodeError) as exc:
                note(f"R{current.race_no} result row: {exc!r} in {line[:24]!r}")
            in_payouts = True
            continue

        if in_payouts:
            try:
                decoded = line.decode("cp932")
            except UnicodeDecodeError as exc:
                note(f"R{current.race_no} payout decode: {exc!r}")
                continue
            found, last_bet = parse_payout_line(decoded, last_bet)
            current.payouts.extend(found)

    return races, outcome
