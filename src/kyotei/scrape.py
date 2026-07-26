"""Live pre-race information and real odds from boatrace.jp.

Why this module exists
----------------------
Two of the project's biggest gaps come from the same place: the daily B/K
archives only tell you what was published in the morning and what happened
afterwards. They contain no odds a bettor could have acted on, and no 直前情報.

Both are on the website:

  直前情報  /owpc/pc/race/beforeinfo?rno={R}&jcd={jcd}&hd={YYYYMMDD}
            展示タイム, チルト, 当日体重, 調整重量, 部品交換,
            スタート展示 (the exhibition *course* and ST), and the water-surface
            conditions -- including 気温 and 水温, which the K feed omits entirely.

  3連単オッズ /owpc/pc/race/odds3t?rno={R}&jcd={jcd}&hd={YYYYMMDD}
            締切時オッズ for all 120 combinations: the actual market price.

What that changes, and what it does not
---------------------------------------
Real odds turn the EV filter from a formality into the thing SPEC §1 actually
asks for. With the historical-average price used in backtest.py, EV is
p x constant, so the filter cannot express "this race is mispriced". With real
odds it can.

Retention is limited, though. Verified against dates where our own archive
confirms the venue was racing: 2020-01-01 returns odds, 2015-01-01 and
2016-06-15 return a "no such race" page. So odds support recent seasons only --
enough for validation and test, not for the whole 2015-2023 training span.

Cost. One request per race, so a single day is ~150 requests (a few minutes at
the project's 1 req/s budget), but a full historical sweep is ~53,000 requests
per year. Daily operation is cheap; backfilling odds is not, and should be
scoped to the evaluation period.
"""

from __future__ import annotations

import argparse
import html as html_module
import re
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

BEFOREINFO_URL = "https://www.boatrace.jp/owpc/pc/race/beforeinfo"
ODDS3T_URL = "https://www.boatrace.jp/owpc/pc/race/odds3t"

#: The site rejects a default python-requests agent on some paths.
USER_AGENT = "Mozilla/5.0 (compatible; kyotei-ml research)"

#: Marks a page for a race that does not exist (or whose data is not retained).
NOT_FOUND_MARKERS = ("指定されたレース", "該当するデータ")


class PageUnavailable(RuntimeError):
    """The page exists but carries no data for this race."""


def beforeinfo_url(day: date, stadium_id: int, race_no: int) -> str:
    return f"{BEFOREINFO_URL}?rno={race_no}&jcd={stadium_id:02d}&hd={day:%Y%m%d}"


def odds3t_url(day: date, stadium_id: int, race_no: int) -> str:
    return f"{ODDS3T_URL}?rno={race_no}&jcd={stadium_id:02d}&hd={day:%Y%m%d}"


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------


def _strip_scripts(markup: str) -> str:
    return re.sub(r"(?s)<(script|style)\b.*?</\1>", " ", markup)


def _text(markup: str) -> str:
    return html_module.unescape(re.sub(r"<[^>]+>", "", markup)).strip()


def _rows(markup: str) -> list[list[tuple[str, str]]]:
    """Every table row as a list of (cell class attribute, cell text)."""
    out = []
    for row in re.findall(r"(?s)<tr[^>]*>(.*?)</tr>", markup):
        cells = []
        for attrs, body in re.findall(r"(?s)<td([^>]*)>(.*?)</td>", row):
            match = re.search(r'class="([^"]*)"', attrs)
            cells.append((match.group(1) if match else "", _text(body)))
        if cells:
            out.append(cells)
    return out


def _number(text: str) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(match.group()) if match else None


# ---------------------------------------------------------------------------
# 3連単オッズ
# ---------------------------------------------------------------------------


def parse_trifecta_odds(markup: str) -> dict[str, float]:
    """Return {'1-2-3': 33.1, ...} for all 120 combinations.

    Table shape: six column groups, one per first-place boat in order 1..6.
    Within a group a cell carrying ``is-fs14`` opens a new second-place block
    (it spans four rows), the next boat-coloured cell is the third-place boat,
    and ``oddsPoint`` closes the triple. Walking cells left to right and
    incrementing the group on each odds cell reconstructs the grid without
    having to model rowspans.
    """
    body = _strip_scripts(markup)
    if any(marker in body for marker in NOT_FOUND_MARKERS):
        raise PageUnavailable("no odds published for this race")

    second_by_group: dict[int, int] = {}
    odds: dict[str, float] = {}

    for cells in _rows(body):
        if not any("oddsPoint" in cls for cls, _ in cells):
            continue  # header rows, the race-number strip, etc.
        group = 0
        third: int | None = None
        for cls, text in cells:
            if "oddsPoint" in cls:
                second = second_by_group.get(group)
                value = _number(text)
                if second is not None and third is not None and value is not None:
                    odds[f"{group + 1}-{second}-{third}"] = value
                group += 1
                third = None
            elif "is-fs14" in cls and "boatColor" in cls:
                if text.isdigit():
                    second_by_group[group] = int(text)
            elif "boatColor" in cls and text.isdigit():
                third = int(text)

    if not odds:
        raise PageUnavailable("odds table present but nothing parsed")
    return odds


def odds_to_market_probabilities(odds: dict[str, float], payback: float = 0.75) -> dict[str, float]:
    """Implied market probability per combination.

    Odds are decimal and already include the takeout, so the raw reciprocals sum
    to roughly 1/payback rather than to 1. Scaling by ``payback`` recovers the
    market's view of the probability, which is what a model has to beat.
    """
    return {
        combination: payback / price
        for combination, price in odds.items()
        if price and price > 0
    }


# ---------------------------------------------------------------------------
# 直前情報
# ---------------------------------------------------------------------------


@dataclass
class BeforeInfo:
    """The 直前情報 a bettor can see roughly 15 minutes before the deadline."""

    race_date: date
    stadium_id: int
    race_no: int
    air_temp_c: float | None = None
    water_temp_c: float | None = None
    wind_speed_m: float | None = None
    wave_height_cm: float | None = None
    weather: str | None = None
    # per lane (1-6)
    exhibition_time: dict[int, float] = field(default_factory=dict)
    tilt: dict[int, float] = field(default_factory=dict)
    weight_kg: dict[int, float] = field(default_factory=dict)
    #: Exhibition start: the course each boat took in the start display, which is
    #: the single best pre-race signal for the course it will actually take.
    exhibition_course: dict[int, int] = field(default_factory=dict)
    exhibition_start: dict[int, float] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return len(self.exhibition_time) == 6


WEATHER_WORDS = ("晴", "曇り", "曇", "雨", "雪", "霧")


def parse_beforeinfo(markup: str, day: date, stadium_id: int, race_no: int) -> BeforeInfo:
    """Parse the 直前情報 page.

    Values are read off the rendered text rather than fixed offsets: this is a
    styled HTML page, not a fixed-width feed, and its column classes are more
    stable than its whitespace.
    """
    body = _strip_scripts(markup)
    if any(marker in body for marker in NOT_FOUND_MARKERS):
        raise PageUnavailable("no 直前情報 published for this race")

    info = BeforeInfo(race_date=day, stadium_id=stadium_id, race_no=race_no)
    flat = re.sub(r"[ \t　]+", " ", html_module.unescape(re.sub(r"<[^>]+>", " ", body)))

    # --- water surface conditions, including the temperatures K omits ---
    for pattern, attribute in (
        (r"気温\s*(-?\d+(?:\.\d+)?)\s*℃", "air_temp_c"),
        (r"水温\s*(-?\d+(?:\.\d+)?)\s*℃", "water_temp_c"),
        (r"風速\s*(-?\d+(?:\.\d+)?)\s*m", "wind_speed_m"),
        (r"波高\s*(-?\d+(?:\.\d+)?)\s*cm", "wave_height_cm"),
    ):
        match = re.search(pattern, flat)
        if match:
            setattr(info, attribute, float(match.group(1)))

    conditions = flat[flat.find("水面気象情報"):] if "水面気象情報" in flat else flat
    for word in WEATHER_WORDS:
        if word in conditions[:200]:
            info.weather = word
            break

    # --- per-boat exhibition table ---
    for cells in _rows(body):
        texts = [text for _, text in cells]
        if not texts or not texts[0].isdigit():
            continue
        lane = int(texts[0])
        if not 1 <= lane <= 6:
            continue
        weight = next((t for t in texts if t.endswith("kg")), None)
        if weight:
            info.weight_kg[lane] = float(weight.replace("kg", ""))
        # Exhibition time and tilt are the two bare decimals that follow.
        decimals = [
            float(t) for t in texts
            if re.fullmatch(r"-?\d+\.\d+", t)
        ]
        if decimals:
            # Exhibition times sit near 6-8 seconds; tilt is -0.5..3.0.
            times = [d for d in decimals if 5.0 <= d <= 9.0]
            tilts = [d for d in decimals if -1.0 <= d <= 4.0 and d not in times]
            if times:
                info.exhibition_time[lane] = times[0]
            if tilts:
                info.tilt[lane] = tilts[0]

    _parse_start_exhibition(flat, info)
    return info


def _parse_start_exhibition(flat: str, info: BeforeInfo) -> None:
    """Read the start-display block: which course each boat took, and its ST.

    The block renders as course/boat/ST triples, e.g. '1 .27  2 .04 ...'. The
    course is the position in the display, and the number before the ST is the
    boat that took it.
    """
    if "スタート展示" not in flat:
        return
    segment = flat[flat.find("スタート展示"):]
    segment = segment[: segment.find("水面気象情報")] if "水面気象情報" in segment else segment
    pairs = re.findall(r"(\d)\s*\.(\d{2})", segment)
    for course, (boat, hundredths) in enumerate(pairs[:6], start=1):
        lane = int(boat)
        if 1 <= lane <= 6:
            info.exhibition_course[lane] = course
            info.exhibition_start[lane] = float(f"0.{hundredths}")


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def fetch_page(session, url: str, *, sleeper=None, timeout: int = 45) -> str:
    if sleeper is not None:
        sleeper()
    response = session.get(url, timeout=timeout)
    if response.status_code != 200:
        raise PageUnavailable(f"HTTP {response.status_code} for {url}")
    content = getattr(response, "content", None)
    return content.decode("utf-8", errors="replace") if content else response.text


def fetch_trifecta_odds(session, day: date, stadium_id: int, race_no: int, *, sleeper=None):
    return parse_trifecta_odds(
        fetch_page(session, odds3t_url(day, stadium_id, race_no), sleeper=sleeper)
    )


def fetch_beforeinfo(session, day: date, stadium_id: int, race_no: int, *, sleeper=None):
    return parse_beforeinfo(
        fetch_page(session, beforeinfo_url(day, stadium_id, race_no), sleeper=sleeper),
        day,
        stadium_id,
        race_no,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    from kyotei.download import RateLimiter, SessionPool
    from kyotei.predict import resolve_stadium

    parser = argparse.ArgumentParser(description="Fetch live odds and 直前情報")
    parser.add_argument("--date", type=date.fromisoformat, default=date.today())
    parser.add_argument("--stadium", required=True, help="code or name, e.g. 24 / 大村")
    parser.add_argument("--race", type=int, required=True)
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--save", default=None, help="write the raw HTML here")
    args = parser.parse_args(argv)

    stadium = resolve_stadium(args.stadium)
    session = SessionPool()
    limiter = RateLimiter(args.rate)

    print(f"=== {args.date} stadium {stadium} {args.race}R ===")

    try:
        info = fetch_beforeinfo(session, args.date, stadium, args.race, sleeper=limiter.acquire)
    except PageUnavailable as exc:
        print(f"直前情報: {exc}")
    else:
        print("\n直前情報")
        print(f"  気温 {info.air_temp_c}℃  水温 {info.water_temp_c}℃  "
              f"風速 {info.wind_speed_m}m  波高 {info.wave_height_cm}cm  {info.weather}")
        for lane in range(1, 7):
            print(
                f"  {lane}号艇 展示 {info.exhibition_time.get(lane)}  "
                f"チルト {info.tilt.get(lane)}  体重 {info.weight_kg.get(lane)}  "
                f"展示進入 {info.exhibition_course.get(lane)}  "
                f"展示ST {info.exhibition_start.get(lane)}"
            )

    try:
        odds = fetch_trifecta_odds(session, args.date, stadium, args.race, sleeper=limiter.acquire)
    except PageUnavailable as exc:
        print(f"\nオッズ: {exc}")
        return 0

    market = odds_to_market_probabilities(odds)
    print(f"\n3連単オッズ: {len(odds)}/120 通り")
    print(f"  市場確率の合計: {sum(market.values()):.4f} (控除率込みで約1.0になるはず)")
    ranked = sorted(odds.items(), key=lambda kv: kv[1])
    print("  人気上位10:")
    for combination, price in ranked[:10]:
        print(f"    {combination}  {price:8.1f}倍   市場確率 {market[combination]:.4%}")

    if args.save:
        Path(args.save).write_text(
            fetch_page(session, odds3t_url(args.date, stadium, args.race),
                       sleeper=limiter.acquire),
            encoding="utf-8",
        )
        print(f"  raw HTML: {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
