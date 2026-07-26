"""Training and probability estimation (Phases 3 and 4).

Phase 3 is the binary "does lane 1 win" baseline: a clean label with a ~55%
base rate, useful because it is easy to sanity-check and hard to get subtly
wrong.

Phase 4 is the real estimator. A per-boat model produces a score, and the
probability of winning is the *race-internal* softmax of those scores
(SPEC §2.3):

    P(i first) = exp(s_i) / Σ_j exp(s_j)

Independent sigmoids would not sum to 1 and every expected-value calculation
downstream would be wrong. From the six first-place probabilities the 120
trifecta probabilities follow by Plackett-Luce sampling without replacement:

    P(i→j→k) = P(i) · P(j)/(1-P(i)) · P(k)/(1-P(i)-P(j))

which is why this is never framed as a 120-class classification.

Splits are strictly chronological (SPEC §2.2). The test years are loaded by
one function that refuses to run unless explicitly asked, so thresholds cannot
drift onto them by accident.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from itertools import permutations
from pathlib import Path

import numpy as np
import polars as pl

from kyotei import features as ft

RACE_KEYS = ["race_date", "stadium_id", "race_no"]

TRAIN_YEARS = range(2015, 2024)   # 2015-2023
VALID_YEARS = range(2024, 2025)   # 2024
TEST_YEARS = range(2025, 2027)    # 2025-2026, touched once

#: All 120 ordered lane triples.
TRIFECTA_COMBINATIONS = tuple(
    f"{a}-{b}-{c}" for a, b, c in permutations(range(1, 7), 3)
)

LGB_PARAMS = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": 42,
    "num_threads": 0,
}


# ---------------------------------------------------------------------------
# Probability machinery -- pure functions, unit-tested without a model
# ---------------------------------------------------------------------------


def race_softmax(scores: np.ndarray) -> np.ndarray:
    """Softmax over one race's boats. Shift-invariant for numerical safety."""
    scores = np.asarray(scores, dtype=np.float64)
    finite = np.where(np.isfinite(scores), scores, -np.inf)
    if not np.any(np.isfinite(finite)):
        return np.full(scores.shape, 1.0 / scores.size)
    shifted = finite - np.max(finite[np.isfinite(finite)])
    exp = np.exp(shifted)
    total = exp.sum()
    if total <= 0 or not np.isfinite(total):
        return np.full(scores.shape, 1.0 / scores.size)
    return exp / total


def normalise_by_race(df: pl.DataFrame, score_column: str = "score") -> pl.DataFrame:
    """Add ``p_win``: the race-internal softmax of ``score_column``.

    Guarantees Σ p_win == 1 within every race key, which is the invariant the
    expected-value calculation depends on.
    """
    # polars 1.4x は window の中に window を書けるが、1.3x は
    # InvalidOperationError: window expression not allowed in aggregation で落ちる。
    # 指数を先に材料化してから合計を取れば、window の入れ子が消えてどちらでも通る。
    max_score = pl.col(score_column).max().over(RACE_KEYS)
    materialised = df.with_columns(
        (pl.col(score_column) - max_score).exp().alias("_exp")
    )
    total = pl.col("_exp").sum().over(RACE_KEYS)
    return materialised.with_columns((pl.col("_exp") / total).alias("p_win")).drop(
        "_exp"
    )


def plackett_luce_trifecta(p_win: dict[int, float] | np.ndarray) -> dict[str, float]:
    """Turn six first-place probabilities into 120 ordered-triple probabilities.

    Sampling without replacement: after boat i is removed, the remaining
    probabilities are renormalised over the survivors.
    """
    if isinstance(p_win, dict):
        probs = np.array([p_win[lane] for lane in range(1, 7)], dtype=np.float64)
    else:
        probs = np.asarray(p_win, dtype=np.float64)
    if probs.shape != (6,):
        raise ValueError(f"expected 6 probabilities, got {probs.shape}")
    if probs.min() < 0:
        raise ValueError("probabilities must be non-negative")

    total = probs.sum()
    if total <= 0:
        raise ValueError("probabilities must not all be zero")
    probs = probs / total

    out: dict[str, float] = {}
    for i, j, k in permutations(range(6), 3):
        remaining_after_i = 1.0 - probs[i]
        remaining_after_j = remaining_after_i - probs[j]
        if remaining_after_i <= 0 or remaining_after_j <= 0:
            out[f"{i + 1}-{j + 1}-{k + 1}"] = 0.0
            continue
        out[f"{i + 1}-{j + 1}-{k + 1}"] = (
            probs[i] * (probs[j] / remaining_after_i) * (probs[k] / remaining_after_j)
        )
    return out


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


@dataclass
class Calibrator:
    """Isotonic map from raw softmax probability to observed frequency.

    A race-internal softmax sums to 1 by construction but is not automatically
    *calibrated*: on real data the raw model is markedly overconfident at the
    top end (it says 75% where the true frequency is 64%). Since the whole
    strategy is "bet when our probability exceeds the market's", overconfidence
    translates directly into over-betting, so it has to be corrected.

    Stored as interpolation knots rather than a pickled estimator: the knots are
    inspectable, survive a scikit-learn upgrade, and need only numpy to apply.

    Applying a monotone map breaks the sum-to-one property, so ``apply``
    renormalises within each race afterwards. Renormalising preserves the
    ordering the isotonic fit produced.
    """

    x: list[float]
    y: list[float]

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        return np.interp(
            np.asarray(probabilities, dtype=np.float64), self.x, self.y
        )

    def apply(self, df: pl.DataFrame, column: str = "p_win") -> pl.DataFrame:
        calibrated = self.transform(df[column].to_numpy())
        out = df.with_columns(pl.Series("_calibrated", calibrated))
        total = pl.col("_calibrated").sum().over(RACE_KEYS)
        return out.with_columns(
            pl.when(total > 0)
            .then(pl.col("_calibrated") / total)
            .otherwise(1.0 / 6.0)
            .alias(column)
        ).drop("_calibrated")

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, payload: dict) -> "Calibrator":
        return cls(list(payload["x"]), list(payload["y"]))


def fit_calibrator(scored: pl.DataFrame, label: str = "won") -> Calibrator:
    """Fit isotonic regression on a split the model did not learn from.

    Must be fitted on validation, never on training data: training-set
    probabilities are already overfit, so a calibrator fitted there would
    conclude the model needs no correction.
    """
    from sklearn.isotonic import IsotonicRegression

    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic.fit(scored["p_win"].to_numpy(), scored[label].to_numpy())
    return Calibrator(
        [float(v) for v in isotonic.X_thresholds_],
        [float(v) for v in isotonic.y_thresholds_],
    )


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def year_of(df: pl.DataFrame) -> pl.Expr:
    return pl.col("race_date").dt.year()


def split(df: pl.DataFrame, years: range) -> pl.DataFrame:
    return df.filter(year_of(df).is_in(list(years)))


def train_valid(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    return split(df, TRAIN_YEARS), split(df, VALID_YEARS)


def load_test_split(df: pl.DataFrame, *, i_understand_this_is_final: bool = False) -> pl.DataFrame:
    """The 2025-2026 hold-out. Refuses to hand it over casually.

    SPEC §2.2/§5: the test years are looked at once, after thresholds are fixed
    on validation. Requiring an explicit flag makes an accidental peek a
    visible line of code rather than a default.
    """
    if not i_understand_this_is_final:
        raise RuntimeError(
            "The test split (2025-2026) is single-use. Tune on the validation "
            "split instead, and pass i_understand_this_is_final=True only for "
            "the final evaluation."
        )
    return split(df, TEST_YEARS)


# ---------------------------------------------------------------------------
# Race-level hygiene
# ---------------------------------------------------------------------------


def complete_races(df: pl.DataFrame) -> pl.DataFrame:
    """Keep only races with 6 boats and exactly one recorded winner.

    A race where the winner is missing (cancelled, or every boat disqualified)
    cannot contribute to a softmax whose probabilities must sum to 1.
    """
    counts = pl.len().over(RACE_KEYS)
    winners = pl.col("won").sum().over(RACE_KEYS)
    return df.filter((counts == 6) & (winners == 1))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class TrainedModel:
    booster: object
    feature_names: list[str]
    feature_set: str
    metrics: dict = field(default_factory=dict)
    calibrator: "Calibrator | None" = None

    def score(self, df: pl.DataFrame) -> np.ndarray:
        """Raw margin (logit), the quantity the race softmax consumes.

        Fed as a numpy matrix rather than a pandas frame: polars is the project
        dataframe (SPEC §4) and pandas would be an extra dependency purely to
        hand LightGBM data it accepts natively. Nulls become NaN, which
        LightGBM reads as missing.
        """
        matrix = df.select(self.feature_names).to_numpy()
        return self.booster.predict(matrix, raw_score=True)

    def predict_probabilities(
        self, df: pl.DataFrame, *, calibrate: bool = True
    ) -> pl.DataFrame:
        scored = normalise_by_race(df.with_columns(pl.Series("score", self.score(df))))
        if calibrate and self.calibrator is not None:
            scored = self.calibrator.apply(scored)
        return scored

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path))
        path.with_suffix(".meta.json").write_text(
            json.dumps(
                {
                    "feature_names": self.feature_names,
                    "feature_set": self.feature_set,
                    "calibrator": self.calibrator.to_dict() if self.calibrator else None,
                    "metrics": self.metrics,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


def train(
    train_df: pl.DataFrame,
    valid_df: pl.DataFrame,
    *,
    label: str = "won",
    feature_set: str = "morning",
    num_boost_round: int = 3000,
    early_stopping: int = 100,
    params: dict | None = None,
) -> TrainedModel:
    import lightgbm as lgb

    names = [c for c in ft.feature_columns(feature_set) if c in train_df.columns]
    if not names:
        raise ValueError("no usable feature columns present")

    settings = {**LGB_PARAMS, **(params or {})}
    train_set = lgb.Dataset(
        train_df.select(names).to_numpy(),
        label=train_df[label].to_numpy(),
        feature_name=names,
    )
    valid_set = lgb.Dataset(
        valid_df.select(names).to_numpy(),
        label=valid_df[label].to_numpy(),
        feature_name=names,
        reference=train_set,
    )
    booster = lgb.train(
        settings,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(early_stopping, verbose=False), lgb.log_evaluation(200)],
    )
    return TrainedModel(booster, names, feature_set)


# ---------------------------------------------------------------------------
# Evaluation -- calibration first, accuracy never (SPEC §2.4)
# ---------------------------------------------------------------------------


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p))


def reliability_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pl.DataFrame:
    """Predicted vs observed frequency per probability bin."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = index == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
                "n": int(mask.sum()),
                "mean_predicted": float(p[mask].mean()),
                "observed": float(y[mask].mean()),
                "gap": float(p[mask].mean() - y[mask].mean()),
            }
        )
    return pl.DataFrame(rows)


def plot_reliability(y: np.ndarray, p: np.ndarray, path: Path, title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = reliability_table(y, p)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    axes[0].plot(table["mean_predicted"], table["observed"], "o-", label="model")
    axes[0].set_xlabel("predicted probability")
    axes[0].set_ylabel("observed frequency")
    axes[0].set_title(f"Reliability: {title}")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(p, bins=30)
    axes[1].set_xlabel("predicted probability")
    axes[1].set_ylabel("count")
    axes[1].set_title("Prediction distribution")
    axes[1].grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def evaluate(
    model: TrainedModel, df: pl.DataFrame, label: str, name: str, reports_dir: Path
) -> dict:
    """Report calibration and ranking quality. Accuracy is deliberately absent."""
    scored = model.predict_probabilities(df)
    y = scored[label].to_numpy()
    p = scored["p_win"].to_numpy()

    metrics = {
        "split": name,
        "rows": int(scored.height),
        "races": int(scored.select(RACE_KEYS).unique().height),
        "base_rate": float(y.mean()),
        "brier": brier_score(y, p),
        "roc_auc": roc_auc(y, p),
        "mean_predicted": float(p.mean()),
    }
    image = plot_reliability(y, p, reports_dir / f"reliability_{name}.png", name)
    metrics["reliability_image"] = str(image)
    metrics["reliability_table"] = reliability_table(y, p).to_dicts()
    return metrics


# ---------------------------------------------------------------------------
# Walk-forward (SPEC §2.2)
# ---------------------------------------------------------------------------


def walk_forward(
    df: pl.DataFrame,
    *,
    first_eval_year: int = 2018,
    last_eval_year: int = 2024,
    feature_set: str = "morning",
    reports_dir: Path | None = None,
) -> list[dict]:
    """Retrain on everything before year Y, evaluate on year Y."""
    out = []
    for year in range(first_eval_year, last_eval_year + 1):
        past = df.filter(year_of(df) < year)
        target = df.filter(year_of(df) == year)
        if past.height < 10_000 or target.is_empty():
            continue
        # The most recent past year acts as the early-stopping set, so the
        # evaluated year is never seen during fitting.
        holdout = past.filter(year_of(past) == year - 1)
        fit = past.filter(year_of(past) < year - 1)
        if fit.is_empty() or holdout.is_empty():
            continue
        model = train(fit, holdout, feature_set=feature_set)
        scored = model.predict_probabilities(target)
        out.append(
            {
                "eval_year": year,
                "train_rows": int(fit.height),
                "rows": int(target.height),
                "brier": brier_score(scored["won"].to_numpy(), scored["p_win"].to_numpy()),
                "roc_auc": roc_auc(scored["won"].to_numpy(), scored["p_win"].to_numpy()),
                "best_iteration": int(model.booster.best_iteration or 0),
            }
        )
        print(f"  walk-forward {year}: {out[-1]}", flush=True)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    from kyotei.paths import PARQUET_DIR, REPORTS_DIR

    parser = argparse.ArgumentParser(description="Train the boat-win model")
    parser.add_argument("--features", default=str(PARQUET_DIR / "features.parquet"))
    parser.add_argument("--reports", default=str(REPORTS_DIR))
    parser.add_argument("--model-out", default=str(PARQUET_DIR / "model.txt"))
    parser.add_argument(
        "--feature-set",
        choices=["morning", "prerace", "realised"],
        default="morning",
        help="morning = B feed only (what a daily run really has); "
             "prerace adds 直前情報; realised adds the actual course (diagnostic)",
    )
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--lane1-baseline", action="store_true",
                        help="also report the Phase 3 'does lane 1 win' baseline")
    args = parser.parse_args(argv)

    reports = Path(args.reports)
    frame = pl.read_parquet(args.features)
    print(f"loaded {frame.height} entry rows")

    usable = complete_races(frame)
    print(f"usable  {usable.height} rows in complete 6-boat races "
          f"({usable.height / max(frame.height, 1):.1%})")

    train_df, valid_df = train_valid(usable)
    print(f"train   {train_df.height} rows ({sorted(TRAIN_YEARS)[0]}-{sorted(TRAIN_YEARS)[-1]})")
    print(f"valid   {valid_df.height} rows (2024)")
    if train_df.is_empty() or valid_df.is_empty():
        print("!! not enough data for the configured splits")
        return 1

    if args.lane1_baseline:
        print("\n=== Phase 3: does lane 1 win? ===")
        lane1_train = train_df.filter(pl.col("lane") == 1)
        lane1_valid = valid_df.filter(pl.col("lane") == 1)
        baseline = train(lane1_train, lane1_valid, feature_set=args.feature_set)
        scored = baseline.booster.predict(lane1_valid.select(baseline.feature_names).to_numpy())
        y = lane1_valid["won"].to_numpy()
        print(f"  base rate (lane 1 wins) : {y.mean():.4f}")
        print(f"  Brier                   : {brier_score(y, scored):.5f}")
        print(f"  Brier of always-base    : {brier_score(y, np.full_like(y, y.mean(), dtype=float)):.5f}")
        print(f"  ROC-AUC                 : {roc_auc(y, scored):.4f}")
        plot_reliability(y, scored, reports / "reliability_lane1.png", "lane1")
        print(reliability_table(y, scored))

    print("\n=== Phase 4: six-boat conditional softmax ===")
    model = train(train_df, valid_df, feature_set=args.feature_set)

    raw = model.predict_probabilities(valid_df, calibrate=False)
    raw_brier = brier_score(raw["won"].to_numpy(), raw["p_win"].to_numpy())
    print("\n  before calibration:")
    print(f"    Brier      : {raw_brier:.5f}")
    print(reliability_table(raw["won"].to_numpy(), raw["p_win"].to_numpy()))

    # Fitted on validation, which the booster only saw for early stopping, so
    # the reported post-calibration figures on this same split are optimistic;
    # the honest number is the one from the test split in backtest.py.
    model.calibrator = fit_calibrator(raw)
    model.metrics["valid"] = evaluate(model, valid_df, "won", "valid", reports)
    print(f"\n  calibration knots: {len(model.calibrator.x)}")

    metrics = model.metrics["valid"]
    print(f"  rows/races : {metrics['rows']} / {metrics['races']}")
    print(f"  base rate  : {metrics['base_rate']:.4f}  (1/6 = {1/6:.4f})")
    print(f"  Brier      : {metrics['brier']:.5f}")
    print(f"  ROC-AUC    : {metrics['roc_auc']:.4f}")
    print(f"  mean p_win : {metrics['mean_predicted']:.4f}")
    print(f"  reliability: {metrics['reliability_image']}")
    print(pl.DataFrame(metrics["reliability_table"]))

    print("\n  top 20 features by gain:")
    gains = sorted(
        zip(model.feature_names, model.booster.feature_importance("gain")),
        key=lambda kv: -kv[1],
    )
    for name, gain in gains[:20]:
        print(f"    {name:28s} {gain:12.0f}")

    if args.walk_forward:
        print("\n=== walk-forward ===")
        model.metrics["walk_forward"] = walk_forward(
            usable, feature_set=args.feature_set, reports_dir=reports
        )

    model.save(Path(args.model_out))
    print(f"\nmodel saved: {args.model_out}")
    (reports / "model_metrics.json").write_text(
        json.dumps(model.metrics, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
