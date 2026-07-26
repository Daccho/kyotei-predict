"""Per-year parse accounting.

SPEC §3.2 warns that the fixed-width layout changed in some periods, and Phase 1
requires the parse success rate reported *by year* so such a change shows up as
a cliff in one year rather than as a vague overall percentage.

SPEC §7 forbids swallowing parse errors, so failures are counted and sampled
here instead of being caught and dropped.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date

#: Per year, keep at most this many example errors. Enough to diagnose a layout
#: change without turning the report into a log dump.
MAX_SAMPLES_PER_YEAR = 5


@dataclass
class FileOutcome:
    """Result of parsing one source file."""

    source: str
    day: date | None
    records_ok: int = 0
    records_failed: int = 0
    fatal: str | None = None  # set when the file could not be read at all
    samples: list[str] = field(default_factory=list)

    @property
    def year(self) -> int | None:
        return self.day.year if self.day else None

    @property
    def ok(self) -> bool:
        return self.fatal is None and self.records_failed == 0


@dataclass
class YearStat:
    files: int = 0
    files_fatal: int = 0
    records_ok: int = 0
    records_failed: int = 0
    samples: list[str] = field(default_factory=list)

    @property
    def records_total(self) -> int:
        return self.records_ok + self.records_failed

    @property
    def record_success_rate(self) -> float:
        return self.records_ok / self.records_total if self.records_total else 1.0

    @property
    def file_success_rate(self) -> float:
        return (self.files - self.files_fatal) / self.files if self.files else 1.0


class ParseStats:
    """Accumulates FileOutcomes and renders the by-year table."""

    def __init__(self, label: str = "") -> None:
        self.label = label
        self.outcomes: list[FileOutcome] = []

    def add(self, outcome: FileOutcome) -> None:
        self.outcomes.append(outcome)

    # -- aggregation --------------------------------------------------------

    def by_year(self) -> dict[int | None, YearStat]:
        out: dict[int | None, YearStat] = defaultdict(YearStat)
        for o in self.outcomes:
            stat = out[o.year]
            stat.files += 1
            stat.records_ok += o.records_ok
            stat.records_failed += o.records_failed
            if o.fatal:
                stat.files_fatal += 1
                if len(stat.samples) < MAX_SAMPLES_PER_YEAR:
                    stat.samples.append(f"{o.source}: {o.fatal}")
            for sample in o.samples:
                if len(stat.samples) < MAX_SAMPLES_PER_YEAR:
                    stat.samples.append(f"{o.source}: {sample}")
        return dict(sorted(out.items(), key=lambda kv: (kv[0] is None, kv[0])))

    @property
    def total(self) -> YearStat:
        combined = YearStat()
        for stat in self.by_year().values():
            combined.files += stat.files
            combined.files_fatal += stat.files_fatal
            combined.records_ok += stat.records_ok
            combined.records_failed += stat.records_failed
        return combined

    def worst_year(self) -> tuple[int | None, YearStat] | None:
        """Year with the lowest record success rate, ignoring empty years."""
        candidates = [(y, s) for y, s in self.by_year().items() if s.records_total]
        if not candidates:
            return None
        return min(candidates, key=lambda kv: kv[1].record_success_rate)

    def layout_change_suspects(self, *, threshold: float = 0.99) -> list[int | None]:
        """Years whose record success rate falls below ``threshold``.

        A layout change shows up as one or two bad years, not a uniform dip.
        """
        return [
            y
            for y, s in self.by_year().items()
            if s.records_total and s.record_success_rate < threshold
        ]

    # -- reporting ----------------------------------------------------------

    def render(self, *, show_samples: bool = True) -> str:
        header = (
            f"{'year':>6} {'files':>7} {'fatal':>6} {'records':>10} "
            f"{'failed':>8} {'rec ok %':>9}"
        )
        lines = []
        if self.label:
            lines.append(f"--- {self.label} ---")
        lines += [header, "-" * len(header)]

        for year, stat in self.by_year().items():
            lines.append(
                f"{str(year) if year is not None else '-':>6} "
                f"{stat.files:>7} {stat.files_fatal:>6} {stat.records_total:>10} "
                f"{stat.records_failed:>8} {stat.record_success_rate * 100:>8.3f}%"
            )

        t = self.total
        lines.append("-" * len(header))
        lines.append(
            f"{'ALL':>6} {t.files:>7} {t.files_fatal:>6} {t.records_total:>10} "
            f"{t.records_failed:>8} {t.record_success_rate * 100:>8.3f}%"
        )

        suspects = self.layout_change_suspects()
        if suspects:
            lines.append(
                "WARNING: below-threshold years (possible layout change): "
                + ", ".join(str(y) for y in suspects)
            )
        if show_samples:
            for year, stat in self.by_year().items():
                if stat.samples:
                    lines.append(f"  [{year}] sample errors:")
                    lines += [f"    {s}" for s in stat.samples]
        return "\n".join(lines)
