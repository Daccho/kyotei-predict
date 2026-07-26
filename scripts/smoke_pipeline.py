"""End-to-end smoke run over whatever has been downloaded so far.

The real splits are yearly (train 2015-2023 / valid 2024 / test 2025-2026) and
need the full archive. This script proves the *chain* works -- parse, features,
leak check, train, race-internal softmax, trifecta, EV filter, ROI -- on a
single year by splitting on month instead, so the pipeline can be validated
before the multi-hour download finishes.

It is a smoke test, not a result: one year of data and a month-based split say
nothing about whether the model is any good. Numbers from here must never be
reported as Phase 3-5 outcomes.

    uv run python scripts/smoke_pipeline.py --start 2015-01-01 --end 2015-12-31
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kyotei import backtest as bt  # noqa: E402
from kyotei import export as ex  # noqa: E402
from kyotei import features as ft  # noqa: E402
from kyotei import model as md  # noqa: E402


def month(df: pl.DataFrame) -> pl.Expr:
    return pl.col("race_date").dt.month()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2015, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2015, 12, 31))
    parser.add_argument("--feature-set", default="morning",
                        choices=["morning", "prerace", "realised"])
    args = parser.parse_args(argv)

    print("=" * 70)
    print("SMOKE RUN -- chronological 70/15/15 split of a short window.")
    print("Not a Phase 3-5 result. Do not quote these numbers as outcomes.")
    print("=" * 70)

    print(f"\n[1/6] parsing {args.start} .. {args.end}", flush=True)
    races, entries, payouts, b_stats, k_stats = ex.parse_range(args.start, args.end)
    print(f"  races={races.height} entries={entries.height} payouts={payouts.height}")
    print("\n  B parse by year:")
    print(b_stats.render(show_samples=False))
    print("\n  K parse by year:")
    print(k_stats.render(show_samples=False))
    if entries.is_empty() or payouts.is_empty():
        print("\n  !! need both B and K archives for this window; download more first")
        return 1

    joined = ex.join_entries(races, entries)
    print(f"\n  joined rows: {joined.height} (entries were {entries.height})")

    print("\n[2/6] building features", flush=True)
    built = ft.build(joined)
    if built.height != joined.height:
        print(f"  !! row count changed {joined.height} -> {built.height}")
        return 1
    print(f"  {built.height} rows, {len(built.columns)} columns")

    print("\n[3/6] leak check: edit the final day, verify earlier rows are unchanged",
          flush=True)
    cutoff = built["race_date"].max()
    tampered_input = joined.with_columns(
        pl.when(pl.col("race_date") == cutoff)
        .then(pl.lit(6))
        .otherwise(pl.col("finish_position"))
        .alias("finish_position")
    )
    tampered = ft.build(tampered_input)
    earlier = pl.col("race_date") < cutoff
    columns = [c for c in ft.feature_columns(args.feature_set) if c in built.columns]
    changed = [
        c for c in columns
        if built.filter(earlier)[c].to_list() != tampered.filter(earlier)[c].to_list()
    ]
    if changed:
        print(f"  !! LEAK: {changed}")
        return 1
    print(f"  clean: {len(columns)} features unchanged when the future was edited")

    print("\n[4/6] training on a chronological 70/15/15 split of the window", flush=True)
    usable = md.complete_races(built)
    print(f"  usable rows in complete 6-boat races: {usable.height} "
          f"({usable.height / built.height:.1%})")

    # Split on dates, not rows, so no race straddles two splits -- and always
    # chronologically, never at random (SPEC §2.2).
    days = sorted(usable["race_date"].unique().to_list())
    first_cut = days[int(len(days) * 0.70)]
    second_cut = days[int(len(days) * 0.85)]
    print(f"  train < {first_cut} <= valid < {second_cut} <= held")
    train_df = usable.filter(pl.col("race_date") < first_cut)
    valid_df = usable.filter(
        (pl.col("race_date") >= first_cut) & (pl.col("race_date") < second_cut)
    )
    final_df = usable.filter(pl.col("race_date") >= second_cut)
    print(f"  train={train_df.height} valid={valid_df.height} held={final_df.height}")
    if min(train_df.height, valid_df.height, final_df.height) < 1000:
        print("  !! not enough rows for a meaningful smoke run")
        return 1

    trained = md.train(train_df, valid_df, feature_set=args.feature_set)
    print(f"  best iteration: {trained.booster.best_iteration}")

    print("\n[5/6] calibration on the held-back month", flush=True)
    scored = trained.predict_probabilities(final_df)
    y = scored["won"].to_numpy()
    p = scored["p_win"].to_numpy()
    print(f"  rows={len(y)} races={scored.select(md.RACE_KEYS).unique().height}")
    print(f"  base rate    : {y.mean():.4f}  (1/6 = {1/6:.4f})")
    print(f"  Brier        : {md.brier_score(y, p):.5f}")
    print(f"  Brier @ 1/6  : {md.brier_score(y, y.mean() + 0 * p):.5f}")
    print(f"  ROC-AUC      : {md.roc_auc(y, p):.4f}")
    print("\n  reliability:")
    print(md.reliability_table(y, p))

    totals = (
        scored.group_by(md.RACE_KEYS).agg(pl.col("p_win").sum().alias("total"))["total"]
    )
    print(f"\n  per-race probability sums: min={totals.min():.6f} max={totals.max():.6f}")
    if abs(totals.min() - 1.0) > 1e-6 or abs(totals.max() - 1.0) > 1e-6:
        print("  !! race probabilities do not sum to 1")
        return 1

    print("\n[6/6] EV-filtered backtest on the held-back month", flush=True)
    dividends = payouts.filter(pl.col("bet_type") == "trifecta").select(
        bt.DIVIDEND_COLUMNS
    )
    price_table = bt.price_table_from(dividends)
    print(f"  priced combinations: {price_table.height}/120")

    tickets = bt.race_tickets(scored)
    priced = bt.settle_tickets(
        bt.attach_expected_payout(tickets, price_table), dividends
    ).filter(pl.col("ev").is_not_null())
    print(f"  tickets: {tickets.height} -> priced+settled: {priced.height}")

    table = bt.threshold_table(priced)
    with pl.Config(tbl_rows=20, float_precision=4, tbl_width_chars=200):
        print(table)
    print(bt.CAVEAT)

    print("=" * 70)
    print("Chain works end to end. These numbers are a smoke test, not a result:")
    print("one year, month split, and the price side is a historical average.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
