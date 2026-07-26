"""Tests for the probability machinery (Phase 4 completion conditions).

SPEC Phase 4 requires proof that the six first-place probabilities sum to 1.0
in every race and that the 120 trifecta probabilities sum to 1.0. Both are
asserted here on pure functions, so they hold regardless of what the booster
learns.
"""

from __future__ import annotations

from datetime import date
from itertools import permutations

import numpy as np
import polars as pl
import pytest

from kyotei import model as md


def race_frame(scores: list[float], race_no: int = 1) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "race_date": [date(2020, 1, 1)] * len(scores),
            "stadium_id": [1] * len(scores),
            "race_no": [race_no] * len(scores),
            "lane": list(range(1, len(scores) + 1)),
            "score": scores,
        }
    )


# ---------------------------------------------------------------------------
# race_softmax
# ---------------------------------------------------------------------------


def test_softmax_sums_to_one():
    assert md.race_softmax([1.0, 2.0, 3.0, 0.5, -1.0, 0.0]).sum() == pytest.approx(1.0)


def test_softmax_is_shift_invariant():
    a = md.race_softmax([1.0, 2.0, 3.0])
    b = md.race_softmax([1001.0, 1002.0, 1003.0])
    assert a == pytest.approx(b)


def test_softmax_survives_large_scores_without_overflow():
    p = md.race_softmax([800.0, 900.0, 1000.0, 700.0, 600.0, 500.0])
    assert p.sum() == pytest.approx(1.0)
    assert np.all(np.isfinite(p))


def test_softmax_survives_very_negative_scores():
    p = md.race_softmax([-900.0, -1000.0, -1100.0, -1200.0, -1300.0, -1400.0])
    assert p.sum() == pytest.approx(1.0)


def test_softmax_of_equal_scores_is_uniform():
    p = md.race_softmax([0.0] * 6)
    assert p == pytest.approx(np.full(6, 1 / 6))


def test_softmax_orders_probabilities_like_scores():
    p = md.race_softmax([3.0, 1.0, 2.0])
    assert p[0] > p[2] > p[1]


def test_softmax_falls_back_to_uniform_when_all_scores_are_nan():
    p = md.race_softmax([np.nan] * 6)
    assert p == pytest.approx(np.full(6, 1 / 6))


# ---------------------------------------------------------------------------
# normalise_by_race -- the Phase 4 invariant on real frames
# ---------------------------------------------------------------------------


def test_probabilities_sum_to_one_within_every_race():
    df = pl.concat(
        [race_frame([1.0, 0.5, 0.0, -0.5, -1.0, -1.5], race_no=n) for n in (1, 2, 3)]
    )
    out = md.normalise_by_race(df)
    totals = out.group_by(md.RACE_KEYS).agg(pl.col("p_win").sum().alias("total"))
    assert totals["total"].to_list() == pytest.approx([1.0] * 3)


def test_races_do_not_leak_probability_into_each_other():
    """Race 2 has huge scores; race 1's probabilities must be unaffected."""
    alone = md.normalise_by_race(race_frame([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], race_no=1))
    together = md.normalise_by_race(
        pl.concat(
            [
                race_frame([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], race_no=1),
                race_frame([500.0] * 6, race_no=2),
            ]
        )
    ).filter(pl.col("race_no") == 1)
    assert alone["p_win"].to_list() == pytest.approx(together["p_win"].to_list())


def test_normalise_matches_the_pure_softmax():
    scores = [0.3, -1.2, 2.0, 0.0, 0.7, -0.4]
    out = md.normalise_by_race(race_frame(scores))
    assert out["p_win"].to_list() == pytest.approx(md.race_softmax(scores).tolist())


def test_independent_sigmoids_would_not_sum_to_one():
    """Documents why the softmax exists: the naive alternative is broken."""
    scores = np.array([1.0, 0.5, 0.0, -0.5, -1.0, -1.5])
    sigmoid_total = float(np.sum(1 / (1 + np.exp(-scores))))
    assert abs(sigmoid_total - 1.0) > 0.5
    assert md.race_softmax(scores).sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Plackett-Luce trifecta
# ---------------------------------------------------------------------------


def test_there_are_exactly_120_trifecta_combinations():
    assert len(md.TRIFECTA_COMBINATIONS) == 120
    assert len(set(md.TRIFECTA_COMBINATIONS)) == 120


def test_trifecta_probabilities_sum_to_one():
    p = md.race_softmax([1.2, 0.4, 0.0, -0.3, -0.8, -1.5])
    trifecta = md.plackett_luce_trifecta(p)
    assert len(trifecta) == 120
    assert sum(trifecta.values()) == pytest.approx(1.0)


def test_trifecta_sums_to_one_for_a_uniform_race():
    trifecta = md.plackett_luce_trifecta(np.full(6, 1 / 6))
    assert sum(trifecta.values()) == pytest.approx(1.0)
    # 6*5*4 = 120 equally likely orderings
    assert all(v == pytest.approx(1 / 120) for v in trifecta.values())


def test_trifecta_sums_to_one_for_a_lopsided_race():
    trifecta = md.plackett_luce_trifecta([0.90, 0.04, 0.03, 0.02, 0.007, 0.003])
    assert sum(trifecta.values()) == pytest.approx(1.0)


def test_trifecta_accepts_a_lane_keyed_dict():
    probs = {lane: 1 / 6 for lane in range(1, 7)}
    assert sum(md.plackett_luce_trifecta(probs).values()) == pytest.approx(1.0)


def test_trifecta_marginal_matches_the_win_probability():
    """Summing every triple that starts with lane i must return P(i wins)."""
    p = md.race_softmax([1.5, 0.2, -0.1, -0.4, -0.9, -1.6])
    trifecta = md.plackett_luce_trifecta(p)
    for lane in range(1, 7):
        marginal = sum(v for k, v in trifecta.items() if k.startswith(f"{lane}-"))
        assert marginal == pytest.approx(p[lane - 1])


def test_trifecta_favours_the_strongest_ordering():
    p = md.race_softmax([2.0, 1.0, 0.0, -1.0, -2.0, -3.0])
    trifecta = md.plackett_luce_trifecta(p)
    assert max(trifecta, key=trifecta.get) == "1-2-3"


def test_trifecta_normalises_unnormalised_input():
    a = md.plackett_luce_trifecta([2.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    b = md.plackett_luce_trifecta([0.25, 0.125, 0.125, 0.125, 0.125, 0.125])
    assert a == pytest.approx(b)


def test_trifecta_covers_every_permutation():
    trifecta = md.plackett_luce_trifecta(np.full(6, 1 / 6))
    expected = {f"{a}-{b}-{c}" for a, b, c in permutations(range(1, 7), 3)}
    assert set(trifecta) == expected


def test_trifecta_rejects_wrong_length():
    with pytest.raises(ValueError):
        md.plackett_luce_trifecta([0.5, 0.5])


def test_trifecta_rejects_negative_probabilities():
    with pytest.raises(ValueError):
        md.plackett_luce_trifecta([-0.1, 0.3, 0.2, 0.2, 0.2, 0.2])


def test_trifecta_rejects_all_zero():
    with pytest.raises(ValueError):
        md.plackett_luce_trifecta([0.0] * 6)


def test_trifecta_handles_a_near_certain_winner_without_dividing_by_zero():
    trifecta = md.plackett_luce_trifecta([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert all(np.isfinite(v) for v in trifecta.values())


# ---------------------------------------------------------------------------
# Splits (SPEC §2.2)
# ---------------------------------------------------------------------------


def years_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "race_date": [date(y, 6, 1) for y in range(2015, 2027)],
            "stadium_id": [1] * 12,
            "race_no": list(range(1, 13)),
            "won": [1] * 12,
        }
    )


def test_train_split_is_2015_to_2023():
    train_df, _ = md.train_valid(years_frame())
    assert train_df["race_date"].dt.year().to_list() == list(range(2015, 2024))


def test_valid_split_is_2024_only():
    _, valid_df = md.train_valid(years_frame())
    assert valid_df["race_date"].dt.year().to_list() == [2024]


def test_train_and_valid_do_not_overlap():
    train_df, valid_df = md.train_valid(years_frame())
    assert set(train_df["race_date"].to_list()).isdisjoint(valid_df["race_date"].to_list())


def test_neither_train_nor_valid_touches_the_test_years():
    train_df, valid_df = md.train_valid(years_frame())
    for frame in (train_df, valid_df):
        assert frame.filter(pl.col("race_date").dt.year() >= 2025).is_empty()


def test_test_split_refuses_without_the_explicit_flag():
    with pytest.raises(RuntimeError, match="single-use"):
        md.load_test_split(years_frame())


def test_test_split_returns_2025_and_2026_when_asked_explicitly():
    held = md.load_test_split(years_frame(), i_understand_this_is_final=True)
    assert sorted(held["race_date"].dt.year().unique().to_list()) == [2025, 2026]


# ---------------------------------------------------------------------------
# Race hygiene
# ---------------------------------------------------------------------------


def boats(race_no: int, wins: list[int]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "race_date": [date(2020, 1, 1)] * len(wins),
            "stadium_id": [1] * len(wins),
            "race_no": [race_no] * len(wins),
            "lane": list(range(1, len(wins) + 1)),
            "won": wins,
        }
    )


def test_complete_races_keeps_a_normal_race():
    df = boats(1, [1, 0, 0, 0, 0, 0])
    assert md.complete_races(df).height == 6


def test_complete_races_drops_a_race_with_five_boats():
    df = boats(1, [1, 0, 0, 0, 0])
    assert md.complete_races(df).is_empty()


def test_complete_races_drops_a_race_with_no_winner():
    df = boats(1, [0, 0, 0, 0, 0, 0])
    assert md.complete_races(df).is_empty()


def test_complete_races_drops_a_race_with_two_winners():
    df = boats(1, [1, 1, 0, 0, 0, 0])
    assert md.complete_races(df).is_empty()


def test_complete_races_keeps_good_races_and_drops_bad_ones_together():
    df = pl.concat([boats(1, [1, 0, 0, 0, 0, 0]), boats(2, [0, 0, 0, 0, 0, 0])])
    out = md.complete_races(df)
    assert out["race_no"].unique().to_list() == [1]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_brier_is_zero_for_perfect_predictions():
    assert md.brier_score(np.array([1, 0, 1]), np.array([1.0, 0.0, 1.0])) == 0.0


def test_brier_is_one_for_perfectly_wrong_predictions():
    assert md.brier_score(np.array([1, 0]), np.array([0.0, 1.0])) == pytest.approx(1.0)


def test_brier_of_the_base_rate_equals_the_variance():
    y = np.array([1, 1, 0, 0, 0, 0])
    p = np.full(6, y.mean())
    assert md.brier_score(y, p) == pytest.approx(y.var())


def test_reliability_table_recovers_a_calibrated_model():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 40_000)
    y = (rng.uniform(0, 1, 40_000) < p).astype(int)
    table = md.reliability_table(y, p)
    assert table.height >= 8
    assert max(abs(g) for g in table["gap"].to_list()) < 0.03


def test_reliability_table_exposes_an_overconfident_model():
    rng = np.random.default_rng(1)
    truth = rng.uniform(0, 1, 20_000)
    y = (rng.uniform(0, 1, 20_000) < truth).astype(int)
    overconfident = np.clip(truth * 1.6, 0, 1)
    table = md.reliability_table(y, overconfident)
    assert max(table["gap"].to_list()) > 0.05


def test_reliability_bins_account_for_every_row():
    p = np.linspace(0, 1, 1000)
    y = (p > 0.5).astype(int)
    assert md.reliability_table(y, p)["n"].sum() == 1000


def test_reliability_handles_probabilities_at_the_bounds():
    y = np.array([0, 1])
    p = np.array([0.0, 1.0])
    assert md.reliability_table(y, p)["n"].sum() == 2
