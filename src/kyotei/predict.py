"""Daily inference: today's positive-EV tickets as a Markdown report (Phase 6).

Timing is the whole difficulty here (SPEC §6). Three clocks matter:

  morning      the B feed for the day is published: roster, published form,
               deadlines. This is when the report below is produced.
  ~15 min out  直前情報 publishes the exhibition run and final conditions.
               Those columns exist in this project only via the K feed, i.e.
               after the race, so a morning run genuinely does not have them.
  after        the K feed publishes results and dividends.

So the deployable model is trained on the ``morning`` feature set, and this CLI
refuses to run a model that expects columns the morning cannot supply. Anything
else would score well in backtests and quietly fail in production.

History still comes from the database: every backward-looking feature needs the
races that came before today, which is exactly what the loaded archive holds.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import polars as pl

from kyotei import backtest as bt
from kyotei import features as ft
from kyotei import model as md
from kyotei import parse_b
from kyotei.download import (
    ExtractError,
    Ledger,
    RateLimiter,
    SessionPool,
    Status,
    daily_url,
    extract_bytes,
    fetch,
)
from kyotei.paths import raw_path

STADIUM_NAMES = {
    1: "桐生", 2: "戸田", 3: "江戸川", 4: "平和島", 5: "多摩川", 6: "浜名湖",
    7: "蒲郡", 8: "常滑", 9: "津", 10: "三国", 11: "びわこ", 12: "住之江",
    13: "尼崎", 14: "鳴門", 15: "丸亀", 16: "児島", 17: "宮島", 18: "徳山",
    19: "下関", 20: "若松", 21: "芦屋", 22: "福岡", 23: "唐津", 24: "大村",
}


class MissingCard(RuntimeError):
    """The B feed for the requested day is not published (or no racing)."""


# ---------------------------------------------------------------------------
# Fetching today's card
# ---------------------------------------------------------------------------


def ensure_card(day: date, *, allow_download: bool = True) -> bytes:
    """Return today's decompressed B payload, downloading it if needed."""
    path = raw_path("B", day)
    if not (path.exists() and path.stat().st_size > 0):
        if not allow_download:
            raise MissingCard(f"no local B archive for {day} and downloads are disabled")
        session = SessionPool()
        status, code = fetch(
            session, daily_url("B", day), path, sleeper=RateLimiter(1.0).acquire
        )
        if status is Status.MISSING:
            raise MissingCard(f"no racing published for {day} (HTTP 404)")
        if status is not Status.DOWNLOADED:
            raise MissingCard(f"could not fetch the card for {day}: HTTP {code}")
        ledger = Ledger()
        ledger.record("B", day, status)
        ledger.save()
    try:
        return extract_bytes(path)
    except ExtractError as exc:
        raise MissingCard(f"card for {day} is unreadable: {exc}") from exc


def card_rows(day: date) -> pl.DataFrame:
    """Parse today's B feed into rows shaped like the historical entries table."""
    races, outcome = parse_b.parse_payload(ensure_card(day), day, f"b{day:%y%m%d}")
    if outcome.records_failed:
        print(f"  warning: {outcome.records_failed} entry rows failed to parse")
        for sample in outcome.samples:
            print(f"    {sample}")
    rows = []
    for race in races:
        for entry in race.entries:
            rows.append(
                {
                    "race_date": day,
                    "stadium_id": race.stadium_id,
                    "race_no": race.race_no,
                    "deadline_time": race.deadline_time,
                    "fixed_course": race.fixed_course,
                    "series_day": race.series_day,
                    "distance_m": race.distance_m,
                    # Outcome and 直前情報 columns are genuinely unknown now.
                    "course": None,
                    "finish_position": None,
                    "finish_status": None,
                    "start_timing": None,
                    "exhibition_time": None,
                    "weather": None,
                    "wind_direction": None,
                    "wind_speed_m": None,
                    "wave_height_cm": None,
                    "decision": None,
                    **{k: v for k, v in entry.items() if k != "series_results"},
                }
            )
    if not rows:
        raise MissingCard(f"card for {day} parsed to zero entries")
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def align_to_history(history: pl.DataFrame, today: pl.DataFrame) -> pl.DataFrame:
    """Concatenate history and today's card on a common schema."""
    for column in history.columns:
        if column not in today.columns:
            today = today.with_columns(pl.lit(None).alias(column))
    today = today.select(history.columns)
    return pl.concat([history, today.cast(history.schema)], how="vertical")


def score_day(
    trained: md.TrainedModel,
    history: pl.DataFrame,
    day: date,
) -> pl.DataFrame:
    """Build features over history+today, then keep today's rows only."""
    today = card_rows(day)
    print(f"  card: {today.height} entries in "
          f"{today.select(md.RACE_KEYS).unique().height} races")

    combined = align_to_history(history, today)
    built = ft.build(combined)
    todays = built.filter(pl.col("race_date") == day)
    if todays.height != today.height:
        raise RuntimeError(
            f"feature build changed today's row count: {today.height} -> {todays.height}"
        )

    missing = [c for c in trained.feature_names if c not in todays.columns]
    if missing:
        raise RuntimeError(f"model needs columns the card cannot supply: {missing}")

    return trained.predict_probabilities(todays)


def price_today(scored: pl.DataFrame, price_table: pl.DataFrame) -> pl.DataFrame:
    tickets = bt.race_tickets(scored)
    return bt.attach_expected_payout(tickets, price_table)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def render_report(
    day: date,
    scored: pl.DataFrame,
    priced: pl.DataFrame,
    threshold: float,
    feature_set: str,
) -> str:
    picks = priced.filter(pl.col("ev") >= threshold).sort("ev", descending=True)
    races = scored.select(md.RACE_KEYS).unique().height

    lines = [
        f"# 買い目レポート {day:%Y-%m-%d}",
        "",
        f"- 対象レース数: **{races}**",
        f"- EV閾値: **{threshold:.2f}**",
        f"- 期待値プラスの買い目: **{picks.height}** 点 "
        f"(全 {priced.height} 点中 {picks.height / max(priced.height, 1):.2%})",
        f"- 特徴量セット: `{feature_set}`",
        "",
    ]

    if feature_set == "morning":
        lines += [
            "> 朝の時点で確定している情報（番組表）のみで算出しています。",
            "> 展示タイム・気象は直前情報でしか確定しないため未使用です。",
            "",
        ]

    if picks.is_empty():
        lines += [
            "## 買い目なし",
            "",
            "期待値が閾値を超える組み合わせはありませんでした。",
            "**見送りも戦略です。** 閾値を下げて無理に買うと控除率25%に負けます。",
            "",
        ]
    else:
        lines += [
            "## 買い目（3連単）",
            "",
            "| 場 | R | 組番 | モデル確率 | 想定払戻 | EV |",
            "|---|---:|---|---:|---:|---:|",
        ]
        for row in picks.head(80).iter_rows(named=True):
            lines.append(
                f"| {STADIUM_NAMES.get(row['stadium_id'], row['stadium_id'])} "
                f"| {row['race_no']} | {row['combination']} "
                f"| {row['p_model']:.3%} | {row['expected_payout']:,.0f}円 "
                f"| {row['ev']:.2f} |"
            )
        if picks.height > 80:
            lines.append(f"| … | | 他 {picks.height - 80} 点 | | | |")
        lines.append("")

    lines += ["## 各レースの1着確率", ""]
    for (stadium_id, race_no), group in scored.sort(
        ["stadium_id", "race_no"]
    ).group_by(["stadium_id", "race_no"], maintain_order=True):
        name = STADIUM_NAMES.get(stadium_id, stadium_id)
        parts = " ".join(
            f"{r['lane']}号艇 {r['p_win']:.1%}"
            for r in group.sort("lane").iter_rows(named=True)
        )
        lines.append(f"- **{name} {race_no}R**: {parts}")

    lines += [
        "",
        "---",
        "",
        "### 注意",
        "",
        "- 想定払戻は各組番の**過去平均配当**です。実際の締切前オッズではありません。",
        "  実運用では本物のオッズに差し替える必要があります。",
        "- 舟券の払戻率は75%です。閾値を下げるほど期待値は控除率に負けていきます。",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    from kyotei.load import DEFAULT_DSN
    from kyotei.paths import PARQUET_DIR, REPORTS_DIR

    parser = argparse.ArgumentParser(description="Daily positive-EV betting report")
    parser.add_argument("--date", type=date.fromisoformat, default=date.today())
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--model", default=str(PARQUET_DIR / "model.txt"))
    parser.add_argument("--features", default=str(PARQUET_DIR / "features.parquet"),
                        help="history frame; must not include the target day")
    parser.add_argument("--reports", default=str(REPORTS_DIR))
    parser.add_argument("--threshold", type=float, default=1.20)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args(argv)

    import lightgbm as lgb

    meta_path = Path(args.model).with_suffix(".meta.json")
    if not meta_path.exists():
        print(f"!! no model metadata at {meta_path}; train a model first")
        return 1
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    feature_set = meta.get("feature_set", "morning")
    trained = md.TrainedModel(
        lgb.Booster(model_file=args.model), meta["feature_names"], feature_set
    )

    if feature_set != "morning":
        print(
            f"!! this model uses the '{feature_set}' feature set, which needs "
            "information a morning run does not have "
            "(直前情報 or the realised course). Retrain with "
            "--feature-set morning before using it to place bets."
        )
        return 1

    print(f"=== {args.date} ===", flush=True)
    history = pl.read_parquet(args.features).filter(pl.col("race_date") < args.date)
    print(f"  history: {history.height} entries up to {args.date}")

    try:
        scored = score_day(trained, history, args.date)
    except MissingCard as exc:
        print(f"  {exc}")
        return 0

    print("  loading dividend history for pricing ...", flush=True)
    payouts = ft.load_frame(
        args.dsn,
        f"""
        SELECT race_date, stadium_id, race_no, combination, payout_yen
        FROM payouts
        WHERE bet_type = 'trifecta' AND race_date < DATE '{args.date.isoformat()}'
        """,
    )
    price_table = (
        bt.expected_dividends(payouts)
        .filter(pl.col("expected_payout").is_not_null())
        .group_by("combination")
        .agg(pl.col("expected_payout").last().alias("expected_payout"))
    )
    priced = price_today(scored, price_table).filter(pl.col("ev").is_not_null())
    print(f"  priced tickets: {priced.height}")

    report = render_report(args.date, scored, priced, args.threshold, feature_set)
    out = Path(args.reports) / f"{args.date:%Y-%m-%d}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")

    picks = priced.filter(pl.col("ev") >= args.threshold).height
    print(f"  positive-EV tickets: {picks}")
    print(f"  report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
