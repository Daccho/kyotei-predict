"""Tests for the ROI simulation (Phase 5).

The numbers here are hand-computable, so a regression in the money arithmetic
shows up as a failing equality rather than a plausible-looking ROI.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from kyotei import backtest as bt
from kyotei import model as md


def tickets(
    rows: list[tuple[str, float, float]],
    *,
    winner: str,
    payout: float,
    race_no: int = 1,
    day: date = date(2024, 1, 1),
) -> pl.DataFrame:
    """rows = [(combination, p_model, expected_payout), ...]"""
    return pl.DataFrame(
        {
            "race_date": [day] * len(rows),
            "stadium_id": [1] * len(rows),
            "race_no": [race_no] * len(rows),
            "combination": [r[0] for r in rows],
            "p_model": [r[1] for r in rows],
            "expected_payout": [r[2] for r in rows],
            "winning_combination": [winner] * len(rows),
            "payout_yen": [payout] * len(rows),
        }
    ).with_columns(
        (pl.col("combination") == pl.col("winning_combination")).alias("hit"),
        (pl.col("p_model") * pl.col("expected_payout") / bt.TICKET_YEN).alias("ev"),
    )


# ---------------------------------------------------------------------------
# EV arithmetic
# ---------------------------------------------------------------------------


def test_ev_is_probability_times_dividend_over_stake():
    frame = tickets([("1-2-3", 0.10, 1500.0)], winner="1-2-3", payout=1500.0)
    assert frame["ev"][0] == pytest.approx(1.5)


def test_fair_priced_ticket_has_ev_of_one():
    frame = tickets([("1-2-3", 0.01, 10_000.0)], winner="1-2-3", payout=10_000.0)
    assert frame["ev"][0] == pytest.approx(1.0)


def test_takeout_means_a_market_priced_ticket_has_ev_below_one():
    """At a 25% takeout, betting the market's own probability loses 25%."""
    market_probability = 0.01
    dividend = bt.PAYBACK_RATE / market_probability * bt.TICKET_YEN
    frame = tickets([("1-2-3", market_probability, dividend)], winner="x", payout=0.0)
    assert frame["ev"][0] == pytest.approx(bt.PAYBACK_RATE)


# ---------------------------------------------------------------------------
# Selection: nothing is bought unconditionally (SPEC §2.5)
# ---------------------------------------------------------------------------


def test_only_tickets_above_the_threshold_are_bought():
    frame = tickets(
        [("1-2-3", 0.20, 1000.0), ("1-2-4", 0.01, 1000.0)],  # EV 2.0 and 0.10
        winner="1-2-3",
        payout=1000.0,
    )
    result = bt.simulate(frame, 1.20)
    assert result.bets == 1
    assert result.hits == 1


def test_a_high_threshold_can_buy_nothing_at_all():
    frame = tickets([("1-2-3", 0.05, 1000.0)], winner="1-2-3", payout=1000.0)
    result = bt.simulate(frame, 5.0)
    assert result.bets == 0
    assert result.roi == 0.0
    assert result.hit_rate == 0.0


def test_raising_the_threshold_never_increases_the_bet_count():
    frame = tickets(
        [(f"1-2-{i}", 0.02 * i, 1000.0) for i in range(3, 7)],
        winner="1-2-3",
        payout=1000.0,
    )
    counts = [bt.simulate(frame, t).bets for t in (0.0, 0.5, 1.0, 1.5)]
    assert counts == sorted(counts, reverse=True)


def test_purchase_rate_is_reported_against_all_available_tickets():
    frame = tickets(
        [("1-2-3", 0.50, 1000.0)] + [(f"1-3-{i}", 0.001, 100.0) for i in range(4, 7)],
        winner="1-2-3",
        payout=1000.0,
    )
    result = bt.simulate(frame, 1.0)
    assert result.total_tickets == 4
    assert result.bets == 1
    assert result.purchase_rate == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Settlement with real dividends
# ---------------------------------------------------------------------------


def test_roi_of_a_single_winning_flat_bet():
    """One 100 yen ticket that pays 1500 yen -> ROI 15.0."""
    frame = tickets([("1-2-3", 0.50, 1500.0)], winner="1-2-3", payout=1500.0)
    result = bt.simulate(frame, 1.0)
    assert result.staked == pytest.approx(100.0)
    assert result.returned == pytest.approx(1500.0)
    assert result.roi == pytest.approx(15.0)


def test_roi_is_zero_when_every_bet_loses():
    frame = tickets([("1-2-3", 0.50, 1500.0)], winner="4-5-6", payout=1500.0)
    result = bt.simulate(frame, 1.0)
    assert result.hits == 0
    assert result.returned == 0.0
    assert result.roi == 0.0


def test_roi_across_many_races_uses_real_payouts():
    frames = [
        tickets([("1-2-3", 0.5, 900.0)], winner="1-2-3", payout=900.0, race_no=1),
        tickets([("1-2-3", 0.5, 900.0)], winner="4-5-6", payout=5000.0, race_no=2),
        tickets([("1-2-3", 0.5, 900.0)], winner="1-2-3", payout=700.0, race_no=3),
    ]
    result = bt.simulate(pl.concat(frames), 1.0)
    assert result.bets == 3
    assert result.hits == 2
    assert result.staked == pytest.approx(300.0)
    # Only the winning tickets' own dividends come back: 900 + 700.
    assert result.returned == pytest.approx(1600.0)
    assert result.roi == pytest.approx(1600 / 300)


def test_hit_rate_counts_only_bought_tickets():
    frame = tickets(
        [("1-2-3", 0.5, 900.0), ("6-5-4", 0.001, 100.0)],
        winner="1-2-3",
        payout=900.0,
    )
    result = bt.simulate(frame, 1.0)
    assert result.bets == 1
    assert result.hit_rate == pytest.approx(1.0)


def test_settlement_does_not_fan_out_rows():
    frame = tickets([("1-2-3", 0.1, 900.0), ("1-2-4", 0.1, 900.0)],
                    winner="1-2-3", payout=900.0)
    dividends = pl.DataFrame(
        {
            "race_date": [date(2024, 1, 1)],
            "stadium_id": [1],
            "race_no": [1],
            "combination": ["1-2-3"],
            "payout_yen": [900.0],
        }
    )
    settled = bt.settle_tickets(frame.drop(["winning_combination", "payout_yen", "hit"]), dividends)
    assert settled.height == 2
    assert settled["hit"].sum() == 1
    assert dict(zip(settled["combination"], settled["payout_yen"]))["1-2-4"] == 0.0


def test_settlement_rejects_two_rows_for_the_same_combination():
    """Two dividends for the SAME combination is corrupt, not a dead heat."""
    frame = tickets([("1-2-3", 0.1, 900.0)], winner="1-2-3", payout=900.0)
    duplicated = pl.DataFrame(
        {
            "race_date": [date(2024, 1, 1)] * 2,
            "stadium_id": [1] * 2,
            "race_no": [1] * 2,
            "combination": ["1-2-3", "1-2-3"],
            "payout_yen": [900.0, 1100.0],
        }
    )
    with pytest.raises(ValueError, match="corrupt rather than a dead heat"):
        bt.settle_tickets(
            frame.drop(["winning_combination", "payout_yen", "hit"]), duplicated
        )


def test_dead_heat_pays_both_winning_combinations():
    """同着 publishes two winning trifectas; a ticket on either one wins, and
    each is paid its own dividend."""
    frame = tickets(
        [("1-2-3", 0.1, 900.0), ("1-3-2", 0.1, 900.0), ("4-5-6", 0.1, 900.0)],
        winner="1-2-3",
        payout=900.0,
    ).drop(["winning_combination", "payout_yen", "hit"])
    dividends = pl.DataFrame(
        {
            "race_date": [date(2024, 1, 1)] * 2,
            "stadium_id": [1] * 2,
            "race_no": [1] * 2,
            "combination": ["1-2-3", "1-3-2"],
            "payout_yen": [900.0, 1500.0],
        }
    )
    settled = bt.settle_tickets(frame, dividends)

    assert settled.height == 3, "no fan-out even with two winners"
    paid = dict(zip(settled["combination"], settled["payout_yen"]))
    assert paid["1-2-3"] == 900.0
    assert paid["1-3-2"] == 1500.0
    assert paid["4-5-6"] == 0.0
    assert settled["hit"].sum() == 2


def test_settlement_drops_races_with_no_dividend():
    """A cancelled race cannot pay, so staking on it must not be simulated."""
    frame = pl.concat(
        [
            tickets([("1-2-3", 0.5, 900.0)], winner="1-2-3", payout=900.0, race_no=1),
            tickets([("1-2-3", 0.5, 900.0)], winner="1-2-3", payout=900.0, race_no=2),
        ]
    ).drop(["winning_combination", "payout_yen", "hit"])
    dividends = pl.DataFrame(
        {
            "race_date": [date(2024, 1, 1)],
            "stadium_id": [1],
            "race_no": [1],
            "combination": ["1-2-3"],
            "payout_yen": [900.0],
        }
    )
    settled = bt.settle_tickets(frame, dividends)
    assert settled["race_no"].to_list() == [1]


# ---------------------------------------------------------------------------
# Leak discipline on the price side
# ---------------------------------------------------------------------------


def dividend_history(payouts: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "race_date": [date(2024, 1, 1) + timedelta(days=i) for i in range(len(payouts))],
            "stadium_id": [1] * len(payouts),
            "race_no": [1] * len(payouts),
            "combination": ["1-2-3"] * len(payouts),
            "payout_yen": payouts,
        }
    )


def test_expected_dividend_excludes_the_current_race():
    history = dividend_history([1000.0] * 40 + [99_999.0])
    out = bt.expected_dividends(history)
    # The final, enormous dividend must not price its own ticket.
    assert out["expected_payout"].to_list()[-1] == pytest.approx(1000.0)


def test_expected_dividend_counts_only_prior_observations():
    out = bt.expected_dividends(dividend_history([1000.0] * 5))
    assert out["expected_payout_n"].to_list() == [0, 1, 2, 3, 4]


def test_expected_dividend_falls_back_before_enough_history():
    """With fewer than MIN_PRIOR_OBSERVATIONS the combination's own mean is not
    trusted; the global mean stands in."""
    out = bt.expected_dividends(dividend_history([1000.0, 2000.0, 3000.0]))
    assert out["expected_payout"].to_list()[0] is None  # nothing at all yet
    assert out["expected_payout"].to_list()[2] == pytest.approx(1500.0)


def test_editing_a_future_dividend_does_not_change_earlier_prices():
    base = dividend_history([1000.0] * 41)
    tampered = base.with_columns(
        pl.when(pl.col("race_date") == date(2024, 1, 1) + timedelta(days=40))
        .then(pl.lit(500_000.0))
        .otherwise(pl.col("payout_yen"))
        .alias("payout_yen")
    )
    a = bt.expected_dividends(base)["expected_payout"].to_list()[:40]
    b = bt.expected_dividends(tampered)["expected_payout"].to_list()[:40]
    assert a == b


# ---------------------------------------------------------------------------
# Kelly staking
# ---------------------------------------------------------------------------


def test_kelly_is_zero_for_a_negative_edge():
    assert bt.kelly_fraction(np.array([0.001]), np.array([1000.0]))[0] == 0.0


def test_kelly_is_positive_for_a_real_edge():
    assert bt.kelly_fraction(np.array([0.5]), np.array([1000.0]))[0] > 0.0


def test_quarter_kelly_is_a_quarter_of_full_kelly():
    p, payout = np.array([0.5]), np.array([1000.0])
    full = bt.kelly_fraction(p, payout, divisor=1.0)[0]
    quarter = bt.kelly_fraction(p, payout, divisor=4.0)[0]
    if full < bt.KELLY_CAP:
        assert quarter == pytest.approx(full / 4)
    else:
        assert quarter <= bt.KELLY_CAP


def test_kelly_is_capped():
    assert bt.kelly_fraction(np.array([0.99]), np.array([10_000.0]))[0] <= bt.KELLY_CAP


def test_kelly_handles_a_dividend_of_exactly_the_stake():
    """Net odds of 0 means no possible profit; stake nothing."""
    assert bt.kelly_fraction(np.array([0.9]), np.array([100.0]))[0] == 0.0


def test_kelly_handles_nan_prices():
    assert bt.kelly_fraction(np.array([0.5]), np.array([np.nan]))[0] == 0.0


def test_kelly_staking_scales_bets_by_confidence():
    frame = pl.concat(
        [
            tickets([("1-2-3", 0.60, 1000.0)], winner="1-2-3", payout=1000.0, race_no=1),
            tickets([("1-2-3", 0.20, 1000.0)], winner="1-2-3", payout=1000.0, race_no=2),
        ]
    )
    flat = bt.simulate(frame, 1.0, staking="flat")
    kelly = bt.simulate(frame, 1.0, staking="quarter_kelly")
    assert flat.staked == pytest.approx(200.0)
    assert kelly.staked > flat.staked, "Kelly sizes off the bankroll, not a flat 100"


def test_unknown_staking_scheme_is_rejected():
    frame = tickets([("1-2-3", 0.5, 1000.0)], winner="1-2-3", payout=1000.0)
    with pytest.raises(ValueError, match="unknown staking"):
        bt.simulate(frame, 1.0, staking="martingale")


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------


def test_max_drawdown_of_a_monotonic_rise_is_zero():
    assert bt.max_drawdown(np.array([0, 1, 2, 3])) == 0.0


def test_max_drawdown_measures_peak_to_trough():
    assert bt.max_drawdown(np.array([0, 10, 4, 8, 1])) == pytest.approx(9.0)


def test_max_drawdown_of_an_empty_curve_is_zero():
    assert bt.max_drawdown(np.array([])) == 0.0


def test_losing_streak_produces_a_drawdown():
    frames = [
        tickets([("1-2-3", 0.5, 1000.0)], winner="9-9-9", payout=1000.0, race_no=n)
        for n in range(1, 6)
    ]
    result = bt.simulate(pl.concat(frames), 1.0)
    assert result.max_dd == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# Reporting shape
# ---------------------------------------------------------------------------


def test_threshold_table_covers_every_threshold_and_staking():
    frame = tickets([("1-2-3", 0.5, 1000.0)], winner="1-2-3", payout=1000.0)
    table = bt.threshold_table(frame)
    assert table.height == len(bt.DEFAULT_THRESHOLDS) * 2
    assert set(table["staking"].unique().to_list()) == {"flat", "quarter_kelly"}


def test_threshold_table_reports_purchase_rate_alongside_roi():
    frame = tickets([("1-2-3", 0.5, 1000.0)], winner="1-2-3", payout=1000.0)
    assert {"purchase_rate", "roi", "hit_rate", "max_dd_yen"} <= set(
        bt.threshold_table(frame).columns
    )


def test_yearly_breakdown_splits_by_year():
    frames = [
        tickets([("1-2-3", 0.5, 1000.0)], winner="1-2-3", payout=1000.0,
                day=date(year, 5, 1))
        for year in (2024, 2025)
    ]
    out = bt.yearly_breakdown(pl.concat(frames), 1.0)
    assert out["year"].to_list() == [2024, 2025]


def test_race_tickets_produces_120_rows_per_race():
    scored = pl.DataFrame(
        {
            "race_date": [date(2024, 1, 1)] * 6,
            "stadium_id": [1] * 6,
            "race_no": [1] * 6,
            "lane": list(range(1, 7)),
            "p_win": [0.5, 0.2, 0.1, 0.1, 0.05, 0.05],
        }
    )
    out = bt.race_tickets(scored)
    assert out.height == 120
    assert out["p_model"].sum() == pytest.approx(1.0)


def test_race_tickets_skips_incomplete_races():
    scored = pl.DataFrame(
        {
            "race_date": [date(2024, 1, 1)] * 5,
            "stadium_id": [1] * 5,
            "race_no": [1] * 5,
            "lane": list(range(1, 6)),
            "p_win": [0.4, 0.2, 0.2, 0.1, 0.1],
        }
    )
    assert bt.race_tickets(scored).is_empty()
