"""Generic CP932 fixed-width record machinery.

Why bytes and not str
---------------------
The published layout counts *bytes*, and racer names are padded with 全角
spaces. A 全角 character is 2 bytes in CP932 but 1 character in Python, so
decoding the record before slicing shifts every column to the right of the
first multi-byte field. Every offset here is therefore a byte offset, and
decoding happens per field, after slicing.

Nothing in this module knows the B or K layouts. It only provides the
mechanism; the actual offset tables live in parse_b.py / parse_k.py and come
from https://www.boatrace.jp/owpc/pc/extra/data/layout.html (SPEC §3.2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence

ENCODING = "cp932"

#: Padding removed from decoded text fields: ASCII space, 全角 space, NUL.
_PAD_CHARS = " 　\t\r\n\x00"

FieldKind = Literal["text", "int", "float", "raw"]


class FieldError(ValueError):
    """A single field could not be interpreted. Carries enough context to fix
    the layout table rather than to guess."""

    def __init__(self, field_name: str, raw: bytes, reason: str) -> None:
        self.field_name = field_name
        self.raw = raw
        self.reason = reason
        super().__init__(f"field {field_name!r}: {reason} (raw={raw!r})")


class RecordError(ValueError):
    """A record could not be parsed. Aggregated by ParseStats, never ignored."""


@dataclass(frozen=True)
class Field:
    """One fixed-width field.

    Args:
        name: destination key.
        start: 0-based byte offset within the record.
        length: field width in bytes.
        kind: how to interpret the sliced bytes.
        scale: for ``kind="float"``, divide the integer reading by this. Feeds
            with implied decimals (e.g. a win rate stored as ``612`` meaning
            6.12) set ``scale=100``.
        allow_blank: when True an all-padding field yields None instead of
            raising. Most optional feed columns are blank-padded.
    """

    name: str
    start: int
    length: int
    kind: FieldKind = "text"
    scale: float = 1.0
    allow_blank: bool = True

    @property
    def stop(self) -> int:
        return self.start + self.length

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"{self.name}: start must be >= 0, got {self.start}")
        if self.length <= 0:
            raise ValueError(f"{self.name}: length must be > 0, got {self.length}")
        if self.scale == 0:
            raise ValueError(f"{self.name}: scale must not be 0")


class Layout:
    """An ordered, non-overlapping set of fields."""

    def __init__(self, name: str, fields: Sequence[Field]) -> None:
        if not fields:
            raise ValueError(f"layout {name!r} has no fields")
        self.name = name
        self.fields = tuple(sorted(fields, key=lambda f: f.start))
        self._check_no_overlap()
        self._check_unique_names()

    @property
    def record_length(self) -> int:
        return max(f.stop for f in self.fields)

    def _check_no_overlap(self) -> None:
        for a, b in zip(self.fields, self.fields[1:]):
            if b.start < a.stop:
                raise ValueError(
                    f"layout {self.name!r}: {a.name} [{a.start}:{a.stop}) overlaps "
                    f"{b.name} [{b.start}:{b.stop})"
                )

    def _check_unique_names(self) -> None:
        seen: set[str] = set()
        for f in self.fields:
            if f.name in seen:
                raise ValueError(f"layout {self.name!r}: duplicate field {f.name!r}")
            seen.add(f.name)

    def gaps(self) -> list[tuple[int, int]]:
        """Byte ranges not covered by any field. Useful when transcribing a
        layout table: an unexpected gap usually means a missed column."""
        out: list[tuple[int, int]] = []
        cursor = self.fields[0].start
        for f in self.fields:
            if f.start > cursor:
                out.append((cursor, f.start))
            cursor = max(cursor, f.stop)
        return out

    def parse(self, record: bytes, *, strict: bool = False) -> tuple[dict[str, Any], list[FieldError]]:
        """Slice and convert one record.

        Returns the parsed mapping plus the per-field errors. With
        ``strict=True`` the first error is raised instead of collected --
        used by tests; the batch path collects so one bad column cannot
        discard an otherwise good file.
        """
        if len(record) < self.record_length:
            raise RecordError(
                f"layout {self.name!r} needs {self.record_length} bytes, "
                f"record has {len(record)}"
            )
        out: dict[str, Any] = {}
        errors: list[FieldError] = []
        for f in self.fields:
            chunk = record[f.start : f.stop]
            try:
                out[f.name] = convert(f, chunk)
            except FieldError as exc:
                if strict:
                    raise
                errors.append(exc)
                out[f.name] = None
        return out, errors


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------


def decode(chunk: bytes) -> str:
    """CP932-decode a field, keeping padding."""
    try:
        return chunk.decode(ENCODING)
    except UnicodeDecodeError as exc:
        raise FieldError("<decode>", chunk, f"not valid {ENCODING}: {exc}") from exc


def clean_text(chunk: bytes) -> str:
    """Decode and strip ASCII *and* 全角 padding."""
    return decode(chunk).strip(_PAD_CHARS)


def convert(field: Field, chunk: bytes) -> Any:
    if field.kind == "raw":
        return chunk

    try:
        text = clean_text(chunk)
    except FieldError as exc:
        raise FieldError(field.name, chunk, exc.reason) from exc

    if field.kind == "text":
        return text

    if not text:
        if field.allow_blank:
            return None
        raise FieldError(field.name, chunk, "blank but required")

    if field.kind == "int":
        return _to_int(field, chunk, text)
    if field.kind == "float":
        return _to_float(field, chunk, text)
    raise FieldError(field.name, chunk, f"unknown kind {field.kind!r}")


def _normalise_digits(text: str) -> str:
    """Map 全角 digits/sign to ASCII. Some columns are 全角 in older files."""
    return text.translate(_ZEN_TO_HAN)


_ZEN_TO_HAN = str.maketrans("０１２３４５６７８９＋－．", "0123456789+-.")


def _to_int(field: Field, chunk: bytes, text: str) -> int | None:
    cleaned = _normalise_digits(text).replace(",", "")
    try:
        return int(cleaned)
    except ValueError as exc:
        raise FieldError(field.name, chunk, f"not an integer: {text!r}") from exc


def _to_float(field: Field, chunk: bytes, text: str) -> float | None:
    cleaned = _normalise_digits(text).replace(",", "")
    try:
        value = float(cleaned)
    except ValueError as exc:
        raise FieldError(field.name, chunk, f"not a number: {text!r}") from exc
    return value / field.scale if field.scale != 1.0 else value


# ---------------------------------------------------------------------------
# Record splitting
# ---------------------------------------------------------------------------


def iter_lines(payload: bytes) -> list[bytes]:
    """Split a decompressed payload into logical lines, dropping blank ones.

    The feeds use CRLF; a stray lone CR or LF must not merge two records.
    """
    normalised = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return [line for line in normalised.split(b"\n") if line.strip()]


def find_marker(line: bytes, marker: bytes) -> bool:
    """True when a line carries a structural marker (e.g. a section sentinel)."""
    return marker in line


def byte_slice(line: bytes, start: int, length: int) -> bytes:
    """Explicit byte slice, padding short lines rather than silently truncating."""
    chunk = line[start : start + length]
    if len(chunk) < length:
        chunk = chunk.ljust(length, b" ")
    return chunk


PredicateT = Callable[[bytes], bool]
