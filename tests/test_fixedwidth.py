"""Tests for the CP932 fixed-width machinery.

No B/K offsets appear here: those come from the published layout table and are
not guessed. What is tested is the mechanism the offset tables will run on.
"""

from __future__ import annotations

import pytest

from kyotei.fixedwidth import (
    Field,
    FieldError,
    Layout,
    RecordError,
    byte_slice,
    clean_text,
    convert,
    decode,
    iter_lines,
)

# A synthetic record: 2-byte lane, 12-byte 全角-padded name, 2-byte grade.
NAME_BYTES = "山田太郎　　".encode("cp932")  # 6 chars -> 12 bytes
SYNTHETIC = b"01" + NAME_BYTES + b"A1"

SYNTHETIC_LAYOUT = Layout(
    "synthetic",
    [
        Field("lane", 0, 2, "int"),
        Field("name", 2, 12, "text"),
        Field("grade", 14, 2, "text"),
    ],
)


# ---------------------------------------------------------------------------
# The reason this module slices bytes rather than str
# ---------------------------------------------------------------------------


def test_full_width_name_occupies_two_bytes_per_character():
    assert len("山田太郎　　") == 6
    assert len(NAME_BYTES) == 12


def test_byte_slicing_reads_the_layout_correctly():
    parsed, errors = SYNTHETIC_LAYOUT.parse(SYNTHETIC)
    assert errors == []
    assert parsed == {"lane": 1, "name": "山田太郎", "grade": "A1"}


def test_decoding_before_slicing_would_corrupt_the_columns():
    """Guards the design decision: str offsets do not match the published table."""
    as_text = SYNTHETIC.decode("cp932")
    # The layout says grade lives at bytes [14:16]; in character space that
    # slice falls off the end of the string entirely.
    assert as_text[14:16] != "A1"
    assert len(as_text) == 10  # 2 + 6 + 2 characters, not 16 bytes


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------


def test_clean_text_strips_full_width_and_ascii_padding():
    assert clean_text("　　山田　".encode("cp932")) == "山田"
    assert clean_text(b"  A1  ") == "A1"
    assert clean_text(b"\x00\x00") == ""


def test_text_field_of_only_padding_becomes_empty_string():
    layout = Layout("t", [Field("name", 0, 4, "text")])
    parsed, errors = layout.parse("　　".encode("cp932"))
    assert (parsed["name"], errors) == ("", [])


# ---------------------------------------------------------------------------
# Numeric conversion
# ---------------------------------------------------------------------------


def test_int_field_tolerates_leading_spaces():
    assert convert(Field("x", 0, 4, "int"), b"  12") == 12


def test_float_scale_applies_implied_decimals():
    # a win rate stored as 612 meaning 6.12
    assert convert(Field("rate", 0, 3, "float", scale=100), b"612") == pytest.approx(6.12)


def test_float_without_scale_is_untouched():
    assert convert(Field("t", 0, 4, "float"), b"6.85") == pytest.approx(6.85)


def test_negative_start_timing_survives():
    """A flying start is recorded as a negative ST and must not be clamped."""
    assert convert(Field("st", 0, 3, "float", scale=100), b"-05") == pytest.approx(-0.05)


def test_full_width_digits_are_normalised():
    assert convert(Field("x", 0, 6, "int"), "１２３".encode("cp932")) == 123


def test_blank_numeric_field_is_none_by_default():
    assert convert(Field("x", 0, 4, "int"), b"    ") is None


def test_blank_numeric_field_raises_when_required():
    with pytest.raises(FieldError):
        convert(Field("x", 0, 4, "int", allow_blank=False), b"    ")


def test_non_numeric_int_field_raises_with_context():
    with pytest.raises(FieldError) as excinfo:
        convert(Field("lane", 0, 4, "int"), b"abcd")
    assert "lane" in str(excinfo.value)
    assert "abcd" in str(excinfo.value)


def test_raw_kind_returns_bytes_untouched():
    assert convert(Field("x", 0, 3, "raw"), b" a ") == b" a "


def test_invalid_cp932_bytes_raise_rather_than_silently_replacing():
    with pytest.raises(FieldError):
        decode(b"\x81")


# ---------------------------------------------------------------------------
# Layout validation -- catches transcription mistakes from the layout table
# ---------------------------------------------------------------------------


def test_overlapping_fields_are_rejected():
    with pytest.raises(ValueError, match="overlaps"):
        Layout("bad", [Field("a", 0, 5, "text"), Field("b", 3, 5, "text")])


def test_duplicate_field_names_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        Layout("bad", [Field("a", 0, 2, "text"), Field("a", 2, 2, "text")])


def test_empty_layout_is_rejected():
    with pytest.raises(ValueError):
        Layout("bad", [])


def test_zero_length_field_is_rejected():
    with pytest.raises(ValueError):
        Field("a", 0, 0, "text")


def test_record_length_is_the_furthest_field_end():
    assert SYNTHETIC_LAYOUT.record_length == 16


def test_gaps_reports_uncovered_ranges():
    layout = Layout("g", [Field("a", 0, 2, "text"), Field("b", 5, 2, "text")])
    assert layout.gaps() == [(2, 5)]


def test_no_gaps_when_fields_are_contiguous():
    assert SYNTHETIC_LAYOUT.gaps() == []


def test_short_record_raises_record_error():
    with pytest.raises(RecordError, match="needs 16 bytes"):
        SYNTHETIC_LAYOUT.parse(b"01")


# ---------------------------------------------------------------------------
# Error collection vs strict mode
# ---------------------------------------------------------------------------


def test_bad_field_is_collected_not_raised_by_default():
    layout = Layout("m", [Field("lane", 0, 2, "int"), Field("name", 2, 4, "text")])
    parsed, errors = layout.parse(b"XXabcd")

    assert parsed["lane"] is None
    assert parsed["name"] == "abcd", "one bad column must not discard the good ones"
    assert len(errors) == 1 and errors[0].field_name == "lane"


def test_strict_mode_raises_on_the_first_bad_field():
    layout = Layout("m", [Field("lane", 0, 2, "int")])
    with pytest.raises(FieldError):
        layout.parse(b"XX", strict=True)


# ---------------------------------------------------------------------------
# Line splitting
# ---------------------------------------------------------------------------


def test_iter_lines_handles_crlf_and_drops_blanks():
    assert iter_lines(b"a\r\nb\r\n\r\nc\r\n") == [b"a", b"b", b"c"]


def test_iter_lines_does_not_merge_records_on_lone_cr():
    assert iter_lines(b"a\rb") == [b"a", b"b"]


def test_iter_lines_keeps_leading_whitespace_inside_a_record():
    assert iter_lines(b"  padded  \r\n") == [b"  padded  "]


def test_byte_slice_pads_short_lines_instead_of_truncating():
    assert byte_slice(b"ab", 0, 5) == b"ab   "


def test_byte_slice_reads_the_requested_window():
    assert byte_slice(b"abcdef", 2, 3) == b"cde"
