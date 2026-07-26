"""Return-on-investment simulation (Phase 5).

What the data can and cannot support
------------------------------------
The K feed publishes the dividend only for the combination that actually won.
That is exactly what a *return* calculation needs -- money comes back only from
winning tickets -- so realised ROI is computed from real payouts throughout.

It is not enough to compute an expected value *before* the race, which needs a
price for every combination we might buy. So the price side is modelled: for
each trifecta combination, the expected dividend is the mean dividend that
combination has paid in *earlier* races only (an expanding mean, same
leak-discipline as features.py). Betting then requires

    EV = P_model(combination) × expected_dividend / 100  ≥  threshold

This is an honest market proxy rather than the real board: it prices each lane
combination at its own historical average, so EV > 1 means "the model thinks
this combination is likelier than history says". A live system must swap in the
real pre-race odds feed; see the caveat printed with every report.

SPEC §2.5 forbids simulating the purchase of every race, so nothing here bets
unconditionally: the threshold and the resulting purchase rate are always
reported together.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from kyotei import model as md

#: Dividends are quoted per 100 yen staked.
TICKET_YEN = 100.0
#: Boat racing returns 75% of the pool; 25% is taken out.
PAYBACK_RATE = 0.75

DEFAULT_THRESHOLDS = (1.00, 1.10, 1.20, 1.30)
#: A combination needs some history before its average dividend means anything.
MIN_PRIOR_OBSERVATIONS = 30
#: Cap on the fraction of bankroll a single Kelly bet may take.
KELLY_CAP = 0.02


# ---------------------------------------------------------------------------
# Market model: expected dividend per combination, from the past only
# ---------------------------------------------------------------------------


def expected_dividends(payouts: pl.DataFrame) -> pl.DataFrame:
    """Expanding mean dividend per trifecta combination.

    ``payouts`` must be one row per race with the winning combination and its
    dividend, sorted chronologically. The returned ``expected_payout`` for a
    race uses only races before it, so a backtest never prices a ticket with
    the very result it is about to bet on.
    """
    if payouts.is_empty():
        return payouts.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("expected_payout"),
            pl.lit(0, dtype=pl.Int64).alias("expected_payout_n"),
        )

    frame = payouts.sort(["race_date", "stadium_id", "race_no"])
    yen = pl.col("payout_yen").cast(pl.Float64)
    group = ["combination"]

    prior_sum = yen.cum_sum().over(group) - yen
    prior_n = pl.int_range(pl.len()).over(group)
    # Global fallback for a combination seen too rarely so far.
    global_sum = yen.cum_sum() - yen
    global_n = pl.int_range(pl.len())

    return frame.with_columns(
        prior_n.alias("expected_payout_n"),
        pl.when(prior_n >= MIN_PRIOR_OBSERVATIONS)
        .then(prior_sum / prior_n)
        .when(global_n > 0)
        .then(global_sum / global_n)
        .otherwise(None)
        .alias("expected_payout"),
    )


DIVIDEND_COLUMNS = ["race_date", "stadium_id", "race_no", "combination", "payout_yen"]


def load_trifecta_dividends(
    *, dsn: str | None = None, parquet: str | None = None, before: object = None
) -> pl.DataFrame:
    """Trifecta dividends from parquet when given, otherwise from Postgres.

    The parquet route lets the whole pipeline run without a database, which
    matters because the container holding Postgres is ephemeral while the
    parsed parquet is the artefact worth keeping.
    """
    if parquet:
        frame = pl.read_parquet(parquet)
        if "bet_type" in frame.columns:
            frame = frame.filter(pl.col("bet_type") == "trifecta")
        frame = frame.select(DIVIDEND_COLUMNS)
    elif dsn:
        from kyotei.features import load_frame

        frame = load_frame(
            dsn,
            f"""
            SELECT {', '.join(DIVIDEND_COLUMNS)}
            FROM payouts WHERE bet_type = 'trifecta'
            """,
        )
    else:
        raise ValueError("provide either parquet= or dsn=")

    if before is not None:
        frame = frame.filter(pl.col("race_date") < before)
    return frame


def price_table_from(dividends: pl.DataFrame) -> pl.DataFrame:
    """Latest expanding-mean price per combination, ready to join onto tickets."""
    priced = expected_dividends(dividends)
    return (
        priced.filter(pl.col("expected_payout").is_not_null())
        .group_by("combination")
        .agg(pl.col("expected_payout").last().alias("expected_payout"))
    )


def combination_frequencies(payouts: pl.DataFrame) -> pl.DataFrame:
    """How often each combination has won so far (diagnostic)."""
    return (
        payouts.group_by("combination")
        .agg(
            pl.len().alias("wins"),
            pl.col("payout_yen").mean().alias("mean_payout"),
            pl.col("payout_yen").median().alias("median_payout"),
        )
        .sort("wins", descending=True)
    )


# ---------------------------------------------------------------------------
# Turning boat probabilities into priced tickets
# ---------------------------------------------------------------------------


def race_tickets(scored: pl.DataFrame) -> pl.DataFrame:
    """Explode per-boat probabilities into 120 priced trifecta rows per race.

    Input needs the race keys, ``lane`` and ``p_win``.
    """
    rows: list[dict] = []
    for (race_date, stadium_id, race_no), group in scored.group_by(
        md.RACE_KEYS, maintain_order=True
    ):
        if group.height != 6:
            continue
        by_lane = dict(zip(group["lane"].to_list(), group["p_win"].to_list()))
        if set(by_lane) != set(range(1, 7)):
            continue
        for combination, probability in md.plackett_luce_trifecta(by_lane).items():
            rows.append(
                {
                    "race_date": race_date,
                    "stadium_id": stadium_id,
                    "race_no": race_no,
                    "combination": combination,
                    "p_model": probability,
                }
            )
    return pl.DataFrame(rows)


def settle_tickets(tickets: pl.DataFrame, dividends: pl.DataFrame) -> pl.DataFrame:
    """Pay each ticket its own dividend, or nothing.

    The join is on (race, combination) rather than on the race key, because a
    race can legitimately publish more than one winning trifecta: a dead heat
    (同着) produces two winning combinations, each with its own dividend. Joining
    the "winner" onto all 120 tickets would fan the frame out in exactly that
    case -- which is how this was found.

    Races with no dividend at all (cancelled, or every boat disqualified) are
    dropped rather than settled: staking on a race that cannot pay would
    understate ROI.
    """
    key = [*md.RACE_KEYS, "combination"]
    prices = dividends.select([*key, "payout_yen"])
    if prices.select(key).is_duplicated().any():
        raise ValueError(
            "the dividend table has two rows for the same (race, combination); "
            "that is corrupt rather than a dead heat"
        )

    eligible = tickets.join(dividends.select(md.RACE_KEYS).unique(),
                            on=md.RACE_KEYS, how="inner")
    expected = eligible.height
    settled = eligible.join(prices, on=key, how="left").with_columns(
        pl.col("payout_yen").is_not_null().alias("hit"),
        pl.col("payout_yen").fill_null(0.0).alias("payout_yen"),
    )
    if settled.height != expected:
        raise ValueError(
            f"settlement changed the row count: {expected} -> {settled.height}"
        )
    return settled


def attach_expected_payout(
    tickets: pl.DataFrame, combination_prices: pl.DataFrame
) -> pl.DataFrame:
    """Join a per-combination price table (combination -> expected_payout).

    This is the *proxy* price: one number per combination for all time. It cannot
    express a per-race view, so EV degenerates to p x constant. Prefer
    attach_real_odds when real odds are available.
    """
    out = tickets.join(combination_prices, on="combination", how="left")
    return out.with_columns(
        (pl.col("p_model") * pl.col("expected_payout") / TICKET_YEN).alias("ev")
    )


def attach_real_odds(tickets: pl.DataFrame, odds: pl.DataFrame) -> pl.DataFrame:
    """Price each ticket with the actual 締切時オッズ for its own race.

    Units: the site quotes decimal odds (33.1 means 33.1x the stake) while the K
    feed quotes yen returned per 100 yen staked. They are converted to the feed's
    convention so the rest of the pipeline is unchanged, which makes
    ``EV = p x odds`` exactly.

    Tickets whose race was not fetched are dropped rather than falling back to a
    proxy price: mixing real and proxy prices in one table would make the ROI
    uninterpretable.
    """
    required = {"race_date", "stadium_id", "race_no", "combination", "odds"}
    missing = required - set(odds.columns)
    if missing:
        raise ValueError(f"odds frame is missing columns: {sorted(missing)}")

    key = [*md.RACE_KEYS, "combination"]
    if odds.select(key).is_duplicated().any():
        raise ValueError("the odds frame has two prices for the same (race, combination)")

    before = tickets.height
    priced = tickets.join(odds.select([*key, "odds"]), on=key, how="inner")
    if priced.height > before:
        raise ValueError(f"pricing fanned out {before} tickets to {priced.height}")

    return priced.with_columns(
        (pl.col("odds") * TICKET_YEN).alias("expected_payout"),
        (pl.col("p_model") * pl.col("odds")).alias("ev"),
    )


def load_odds(path: str) -> pl.DataFrame:
    """Read a real-odds parquet produced by odds_backfill."""
    frame = pl.read_parquet(path)
    return frame.with_columns(
        pl.col("race_date").cast(pl.Date),
        pl.col("stadium_id").cast(pl.Int16),
        pl.col("race_no").cast(pl.Int16),
    )


# ---------------------------------------------------------------------------
# Staking
# ---------------------------------------------------------------------------


def kelly_fraction(p: np.ndarray, payout_yen: np.ndarray, *, divisor: float = 4.0) -> np.ndarray:
    """Fractional Kelly stake. ``divisor=4`` gives the quarter-Kelly SPEC asks for.

    Net odds b = payout/100 - 1. f* = (p·b - (1-p)) / b, floored at 0 and
    capped so a single ticket cannot take an outsized share of the bankroll.
    """
    p = np.asarray(p, dtype=np.float64)
    b = np.asarray(payout_yen, dtype=np.float64) / TICKET_YEN - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        full = np.where(b > 0, (p * b - (1.0 - p)) / b, 0.0)
    full = np.nan_to_num(full, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(full / divisor, 0.0, KELLY_CAP)


def max_drawdown(cumulative: np.ndarray) -> float:
    """Largest peak-to-trough fall of a cumulative P&L curve."""
    series = np.asarray(cumulative, dtype=np.float64)
    if series.size == 0:
        return 0.0
    peak = np.maximum.accumulate(series)
    return float(np.max(peak - series))


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


@dataclass
class Result:
    threshold: float
    staking: str
    bets: int
    races_with_a_bet: int
    total_races: int
    total_tickets: int
    hits: int
    staked: float
    returned: float

    @property
    def purchase_rate(self) -> float:
        """Share of all available tickets bought (SPEC §2.5 'purchase rate')."""
        return self.bets / self.total_tickets if self.total_tickets else 0.0

    @property
    def race_participation(self) -> float:
        return self.races_with_a_bet / self.total_races if self.total_races else 0.0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.bets if self.bets else 0.0

    @property
    def roi(self) -> float:
        return self.returned / self.staked if self.staked else 0.0

    max_dd: float = 0.0

    def row(self) -> dict:
        return {
            "threshold": self.threshold,
            "staking": self.staking,
            "bets": self.bets,
            "purchase_rate": self.purchase_rate,
            "races_bet": self.race_participation,
            "hit_rate": self.hit_rate,
            "roi": self.roi,
            "max_dd_yen": self.max_dd,
            "staked_yen": self.staked,
            "returned_yen": self.returned,
        }


def simulate(
    priced: pl.DataFrame,
    threshold: float,
    *,
    staking: str = "flat",
    bankroll: float = 1_000_000.0,
) -> Result:
    """Buy only tickets whose EV clears ``threshold``; settle with real payouts."""
    total_tickets = priced.height
    total_races = priced.select(md.RACE_KEYS).unique().height

    chosen = priced.filter(pl.col("ev") >= threshold).sort(
        ["race_date", "stadium_id", "race_no"]
    )
    if chosen.is_empty():
        return Result(threshold, staking, 0, 0, total_races, total_tickets, 0, 0.0, 0.0)

    hit = chosen["hit"].to_numpy()
    payout = chosen["payout_yen"].to_numpy(allow_copy=True).astype(np.float64)

    if staking == "flat":
        stake = np.full(chosen.height, TICKET_YEN)
    elif staking == "quarter_kelly":
        fraction = kelly_fraction(
            chosen["p_model"].to_numpy(),
            chosen["expected_payout"].to_numpy(allow_copy=True),
        )
        stake = fraction * bankroll
    else:
        raise ValueError(f"unknown staking scheme {staking!r}")

    # Winning tickets return stake/100 x the quoted dividend.
    returned = np.where(hit, stake / TICKET_YEN * payout, 0.0)
    # The equity curve starts at zero, so a run of losses from the very first
    # bet counts as a drawdown instead of being measured from an already-sunk
    # first stake.
    pnl = np.concatenate([[0.0], np.cumsum(returned - stake)])

    return Result(
        threshold=threshold,
        staking=staking,
        bets=int(chosen.height),
        races_with_a_bet=int(chosen.select(md.RACE_KEYS).unique().height),
        total_races=total_races,
        total_tickets=total_tickets,
        hits=int(hit.sum()),
        staked=float(stake.sum()),
        returned=float(returned.sum()),
        max_dd=max_drawdown(pnl),
    )


def yearly_breakdown(priced: pl.DataFrame, threshold: float, *, staking: str = "flat") -> pl.DataFrame:
    """ROI per year, so a strategy that only worked in one year is visible."""
    rows = []
    for year in sorted(priced["race_date"].dt.year().unique().to_list()):
        subset = priced.filter(pl.col("race_date").dt.year() == year)
        rows.append({"year": year, **simulate(subset, threshold, staking=staking).row()})
    return pl.DataFrame(rows)


def threshold_table(
    priced: pl.DataFrame,
    thresholds=DEFAULT_THRESHOLDS,
    stakings=("flat", "quarter_kelly"),
) -> pl.DataFrame:
    return pl.DataFrame(
        [
            simulate(priced, threshold, staking=staking).row()
            for staking in stakings
            for threshold in thresholds
        ]
    )


REAL_ODDS_NOTE = """
Prices are REAL 締切時オッズ
  Both sides of this simulation now come from the market: tickets are priced at
  the odds actually offered at close, and winners are paid the actual dividend.
  Two things still bound the result:
    * coverage -- only the races present in the odds file are simulated, so check
      the covered-races figure above before reading the ROI as a season result;
    * the odds are the closing odds, which is what a bet placed just before the
      deadline gets, not what an earlier bet would have got.
"""

CAVEAT = """
CAVEAT on the price side
  Dividends are real (K feed), but the pre-race price of a ticket we did NOT
  win is not published, so EV uses each combination's expanding historical mean
  dividend as its price. This prices a combination at its own base rate rather
  than at what the market actually offered on the day, so:
    * a positive result means the model beats the historical base rate, which is
      weaker evidence than beating the live board;
    * live deployment must substitute the real odds feed before any real money.
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    from kyotei.load import DEFAULT_DSN
    from kyotei.paths import PARQUET_DIR, REPORTS_DIR

    parser = argparse.ArgumentParser(description="Backtest EV-filtered trifecta betting")
    parser.add_argument("--features", default=str(PARQUET_DIR / "features.parquet"))
    parser.add_argument("--model", default=str(PARQUET_DIR / "model.txt"))
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--payouts", default=None,
                        help="dividends parquet; skips Postgres entirely")
    parser.add_argument("--odds", default=None,
                        help="real 締切時オッズ parquet from odds_backfill. When "
                             "given, tickets are priced per race instead of with "
                             "the historical-average proxy")
    parser.add_argument("--reports", default=str(REPORTS_DIR))
    parser.add_argument(
        "--split",
        choices=["valid", "test"],
        default="valid",
        help="'test' uses the single-use 2025-2026 hold-out",
    )
    parser.add_argument("--final", action="store_true",
                        help="required with --split test (SPEC §5)")
    args = parser.parse_args(argv)

    import lightgbm as lgb
    import json

    frame = md.complete_races(pl.read_parquet(args.features))
    meta = json.loads(Path(args.model).with_suffix(".meta.json").read_text(encoding="utf-8"))
    booster = lgb.Booster(model_file=args.model)
    trained = md.TrainedModel(
        booster, meta["feature_names"], meta.get("feature_set", "morning")
    )
    if meta.get("calibrator"):
        trained.calibrator = md.Calibrator.from_dict(meta["calibrator"])
        print("using the stored isotonic calibrator")
    else:
        print("!! model has no calibrator; raw softmax is overconfident, so EV "
              "will be overstated")

    if args.split == "test":
        subset = md.load_test_split(frame, i_understand_this_is_final=args.final)
    else:
        subset = md.split(frame, md.VALID_YEARS)
    print(f"{args.split} split: {subset.height} rows, "
          f"{subset.select(md.RACE_KEYS).unique().height} races")
    if subset.is_empty():
        print("!! no rows in the chosen split")
        return 1

    print("loading real trifecta dividends ...", flush=True)
    payouts = load_trifecta_dividends(dsn=args.dsn, parquet=args.payouts)
    print(f"dividends: {payouts.height}")


    print("building tickets ...", flush=True)
    scored = trained.predict_probabilities(subset)
    tickets = race_tickets(scored)
    print(f"tickets: {tickets.height} ({tickets.height / 120:.0f} races x 120)")

    if args.odds:
        odds = load_odds(args.odds)
        races_with_odds = odds.select(md.RACE_KEYS).unique().height
        print(f"real odds: {odds.height} prices covering {races_with_odds} races")
        priced = attach_real_odds(tickets, odds)
        covered = priced.select(md.RACE_KEYS).unique().height
        print(f"priced with REAL odds: {priced.height} tickets in {covered} races "
              f"({covered / max(tickets.height // 120, 1):.1%} of the split)")
        using_real_odds = True
    else:
        priced = attach_expected_payout(tickets, price_table_from(payouts))
        print("priced with the historical-average PROXY (no --odds given)")
        using_real_odds = False

    priced = settle_tickets(priced, payouts).filter(pl.col("ev").is_not_null())
    print(f"settled: {priced.height} tickets")

    table = threshold_table(priced)
    print("\n=== EV threshold table ===")
    with pl.Config(tbl_rows=40, float_precision=4, tbl_width_chars=200):
        print(table)

    print("\n=== yearly ROI (flat, threshold 1.20) ===")
    with pl.Config(tbl_rows=20, float_precision=4, tbl_width_chars=200):
        print(yearly_breakdown(priced, 1.20))

    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    table.write_csv(reports / f"backtest_{args.split}.csv")
    print(f"\nwritten: {reports / f'backtest_{args.split}.csv'}")
    print(REAL_ODDS_NOTE if using_real_odds else CAVEAT)

    best = table.filter(pl.col("staking") == "flat").sort("roi", descending=True)
    if best.height and best["roi"][0] > 1.0:
        print("!! ROI above 100%: treat as a suspected leak, not a result "
              "(SPEC §8). Published kyotei models land at 80-95%.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
