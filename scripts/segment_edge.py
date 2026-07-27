"""Where, if anywhere, does the model beat the market?

The pooled result says the market is better calibrated than the model. That is
an average, and an average can hide a segment where the ordering reverses --
thin markets on a quiet weekday are priced by fewer and less informed bets than
an SG final. If such a segment exists, it is the only place worth betting; if
none does, no amount of extra features will change the conclusion, because the
problem is informational rather than statistical.

Method. For each race we hold two full distributions over the 120 trifecta
combinations: the model's, and the market's (1/odds, renormalised to sum to 1 so
the takeout is removed and the two are comparable). Exactly one combination
wins, so the natural score is the log loss of the realised winner,

    -log p(winning combination)

averaged within a segment. Lower is better. ``edge`` below is
market_loss - model_loss, so a positive number means the model predicted that
segment better than the market did.

Log loss rather than Brier: with 120 outcomes, 119 of which are near zero,
Brier is dominated by how confidently we said "no" to combinations nobody
considered, which is not what the betting decision turns on.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kyotei import backtest as bt  # noqa: E402
from kyotei import model as md  # noqa: E402

PARQUET = Path("data/parquet")
FLOOR = 1e-9  # keeps log finite when a model probability underflows


def build_tickets() -> pl.DataFrame:
    import lightgbm as lgb

    feats = md.complete_races(pl.read_parquet(PARQUET / "features.parquet"))
    valid = md.split(feats, md.VALID_YEARS)

    meta = json.loads((PARQUET / "model.meta.json").read_text(encoding="utf-8"))
    trained = md.TrainedModel(
        lgb.Booster(model_file=str(PARQUET / "model.txt")),
        meta["feature_names"],
        meta["feature_set"],
    )
    trained.calibrator = md.Calibrator.from_dict(meta["calibrator"])
    scored = trained.predict_probabilities(valid)

    odds = bt.load_odds("data/odds/odds3t_2024.parquet")
    payouts = bt.load_trifecta_dividends(parquet=str(PARQUET / "payouts.parquet"))
    tickets = bt.settle_tickets(
        bt.attach_real_odds(bt.race_tickets(scored), odds), payouts
    )

    # Market distribution: strip the takeout by renormalising within the race,
    # so model and market are both proper distributions over the same support.
    raw = 1.0 / pl.col("odds")
    tickets = tickets.with_columns(
        (raw / raw.sum().over(md.RACE_KEYS)).alias("p_market")
    )

    # Race attributes to segment on.
    attrs = (
        valid.group_by(md.RACE_KEYS)
        .agg(
            pl.col("series_grade").first(),
            pl.col("fixed_course_flag").first(),
            pl.col("wind_speed_m").first(),
        )
    )
    return tickets.join(attrs, on=md.RACE_KEYS, how="left")


def segment_report(tickets: pl.DataFrame, by: list[str], label: str) -> pl.DataFrame:
    winners = tickets.filter(pl.col("hit"))
    model_loss = -pl.col("p_model").clip(FLOOR, 1.0).log()
    market_loss = -pl.col("p_market").clip(FLOOR, 1.0).log()

    scores = (
        winners.with_columns(
            model_loss.alias("_m"), market_loss.alias("_k")
        )
        .group_by(by)
        .agg(
            pl.len().alias("races"),
            pl.col("_m").mean().alias("model_loss"),
            pl.col("_k").mean().alias("market_loss"),
        )
        .with_columns((pl.col("market_loss") - pl.col("model_loss")).alias("edge"))
    )

    bets = tickets.filter(pl.col("ev") >= 1.2)
    money = (
        bets.group_by(by)
        .agg(
            pl.len().alias("bets"),
            (
                (pl.col("payout_yen") * pl.col("hit")).sum()
                / (bt.TICKET_YEN * pl.len())
            ).alias("roi"),
        )
    )
    out = scores.join(money, on=by, how="left").sort("edge", descending=True)
    print(f"\n=== {label} ===")
    with pl.Config(tbl_rows=30, float_precision=4, tbl_width_chars=200):
        print(out)
    return out


def main() -> int:
    print("building tickets with both distributions ...", flush=True)
    tickets = build_tickets()
    races = tickets.select(md.RACE_KEYS).unique().height
    print(f"{tickets.height} tickets over {races} races")

    winners = tickets.filter(pl.col("hit"))
    m = float((-np.log(np.clip(winners["p_model"].to_numpy(), FLOOR, 1))).mean())
    k = float((-np.log(np.clip(winners["p_market"].to_numpy(), FLOOR, 1))).mean())
    print(f"\npooled log loss   model {m:.4f}   market {k:.4f}   "
          f"edge {k - m:+.4f}  (positive = model better)")
    print(f"a random guess would be {np.log(120):.4f}")

    for by, label in (
        (["stadium_id"], "by stadium"),
        (["series_grade"], "by series grade"),
        (["fixed_course_flag"], "by 進入固定"),
        (["race_no"], "by race number"),
    ):
        segment_report(tickets, by, label)

    print("\n" + "=" * 72)
    best = segment_report(tickets, ["stadium_id"], "ranked by edge").head(3)
    if best["edge"].max() > 0:
        print("Segments where the model beat the market exist -- see the top rows.")
        print("Next step: re-run the backtest restricted to them.")
    else:
        print("The market wins in EVERY segment tested.")
        print("The gap is informational, not statistical: more features on the")
        print("same public inputs will not close it. This is the kill criterion.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
