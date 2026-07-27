"""Train on 2015-2023 and backtest 2024 against the real closing odds.

This is the question the whole project turns on. Every earlier backtest priced
tickets at each combination's historical average, which is one number per
combination for all time, so EV reduced to p x constant and the filter could
only ever mean "buy likely combinations". With real per-race odds it can finally
mean "this race is mispriced".

Coverage caveat: only the races present in the odds file are simulated. The
backfill visits races in a fixed-seed shuffled order precisely so that a partial
file is an unbiased sample of the season rather than its first months.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kyotei import backtest as bt  # noqa: E402
from kyotei import features as ft  # noqa: E402
from kyotei import model as md  # noqa: E402
from kyotei import schedule as sch  # noqa: E402

PARQUET = Path("data/parquet")
ODDS = Path("data/odds/odds3t_2024.parquet")


def main() -> int:
    print("[1/5] loading entries + schedule", flush=True)
    entries = pl.read_parquet(PARQUET / "entries.parquet")
    schedule = pl.read_parquet(PARQUET / "schedule.parquet")
    print(f"  entries {entries.height}  schedule {schedule.height}")

    before = entries.height
    entries = sch.attach_grade(
        entries.with_columns(pl.col("stadium_id").cast(pl.Int16)), schedule
    )
    print(f"  after grade join: {entries.height} (unchanged: {entries.height == before})")
    print(f"  races with a known grade: "
          f"{1 - entries['series_grade'].null_count() / entries.height:.1%}")

    print("\n[2/5] building leak-free features", flush=True)
    built = ft.build(entries)
    if built.height != entries.height:
        print(f"  !! row count changed {entries.height} -> {built.height}")
        return 1
    built.write_parquet(PARQUET / "features.parquet")
    print(f"  {built.height} rows, {len(built.columns)} columns")

    usable = md.complete_races(built)
    print(f"  usable in complete 6-boat races: {usable.height} "
          f"({usable.height / built.height:.1%})")

    print("\n[3/5] training on 2015-2023, early stopping on 2024", flush=True)
    train_df, valid_df = md.train_valid(usable)
    print(f"  train {train_df.height}   valid {valid_df.height}")
    trained = md.train(train_df, valid_df, feature_set="morning")
    print(f"  best iteration {trained.booster.best_iteration}")

    raw = trained.predict_probabilities(valid_df, calibrate=False)
    trained.calibrator = md.fit_calibrator(raw)
    scored = trained.predict_probabilities(valid_df)

    y = scored["won"].to_numpy()
    print(f"  Brier raw        {md.brier_score(y, raw['p_win'].to_numpy()):.5f}")
    print(f"  Brier calibrated {md.brier_score(y, scored['p_win'].to_numpy()):.5f}")
    print(f"  Brier @ 1/6      {md.brier_score(y, y * 0 + 1 / 6):.5f}")
    print(f"  ROC-AUC          {md.roc_auc(y, scored['p_win'].to_numpy()):.4f}")

    print("\n  top 12 features by gain:")
    gains = sorted(zip(trained.feature_names, trained.booster.feature_importance("gain")),
                   key=lambda kv: -kv[1])
    for name, gain in gains[:12]:
        print(f"    {name:26s} {gain:12.0f}")
    trained.save(PARQUET / "model.txt")

    print("\n[4/5] pricing 2024 tickets with REAL closing odds", flush=True)
    odds = bt.load_odds(str(ODDS))
    covered = odds.select(md.RACE_KEYS).unique().height
    print(f"  odds: {odds.height} prices over {covered} races "
          f"({covered / max(valid_df.height // 6, 1):.1%} of the split)")

    tickets = bt.race_tickets(scored)
    payouts = bt.load_trifecta_dividends(parquet=str(PARQUET / "payouts.parquet"))
    real = bt.settle_tickets(bt.attach_real_odds(tickets, odds), payouts)
    print(f"  settled {real.height} tickets in "
          f"{real.select(md.RACE_KEYS).unique().height} races")

    print("\n[5/5] results", flush=True)
    with pl.Config(tbl_rows=20, float_precision=4, tbl_width_chars=200):
        print("\n=== REAL odds ===")
        print(bt.threshold_table(real))

        # Same races, priced the old way, so the comparison is like for like.
        proxy = bt.settle_tickets(
            bt.attach_expected_payout(
                tickets.join(odds.select(md.RACE_KEYS).unique(), on=md.RACE_KEYS),
                bt.price_table_from(payouts),
            ),
            payouts,
        )
        print("\n=== historical-average PROXY, same races ===")
        print(bt.threshold_table(proxy))

    print(bt.REAL_ODDS_NOTE)
    best = bt.threshold_table(real).filter(pl.col("staking") == "flat")
    if best["roi"].max() > 1.0:
        print("!! ROI above 100%: suspect a leak first (SPEC §8).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
