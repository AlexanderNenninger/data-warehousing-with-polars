"""Property-based tests for data_warehousing_with_polars using Hypothesis."""

from __future__ import annotations

import string

import polars as pl
import pytest
from data_warehousing_with_polars.scd import _partition_predicate, _sql_literal
from data_warehousing_with_polars.schema import SchemaError, schema
from hypothesis import given
from hypothesis import strategies as st

# ── Strategies ──────────────────────────────────────────────────────────────────

# Column names: lowercase letters + underscore, no leading underscore.
_col_name = st.text(
    alphabet=string.ascii_lowercase + "_",
    min_size=1,
    max_size=10,
).filter(lambda s: s[0] != "_" and s.strip("_"))


@st.composite
def matching_schema_and_frame(draw):
    """A schema dict and a LazyFrame whose columns match exactly."""
    cols = draw(st.lists(_col_name, min_size=1, max_size=6, unique=True))
    n_rows = draw(st.integers(0, 20))
    expected = {col: pl.Int64 for col in cols}
    data = {
        col: draw(st.lists(st.integers(-1000, 1000), min_size=n_rows, max_size=n_rows))
        for col in cols
    }
    return expected, pl.DataFrame(data, schema=expected).lazy()


@st.composite
def schema_and_frame_with_extras(draw):
    """Schema + LazyFrame with all expected columns PLUS extra columns."""
    base_cols = draw(st.lists(_col_name, min_size=1, max_size=4, unique=True))
    extra_cols = draw(
        st.lists(_col_name, min_size=1, max_size=3, unique=True).filter(
            lambda lst: all(c not in base_cols for c in lst)
        )
    )
    n_rows = draw(st.integers(0, 10))
    expected = {col: pl.Int64 for col in base_cols}
    all_cols = base_cols + extra_cols
    data = {
        col: draw(st.lists(st.integers(-1000, 1000), min_size=n_rows, max_size=n_rows))
        for col in all_cols
    }
    return expected, pl.DataFrame(data, schema={c: pl.Int64 for c in all_cols}).lazy()


@st.composite
def schema_and_frame_with_missing(draw):
    """Schema + LazyFrame that is missing at least one expected column."""
    cols = draw(st.lists(_col_name, min_size=2, max_size=6, unique=True))
    n_rows = draw(st.integers(0, 10))
    expected = {col: pl.Int64 for col in cols}
    # Drop at least one column from the frame.
    n_drop = draw(st.integers(1, len(cols)))
    present_cols = cols[n_drop:]
    data = {
        col: draw(st.lists(st.integers(-1000, 1000), min_size=n_rows, max_size=n_rows))
        for col in present_cols
    }
    return expected, pl.DataFrame(data, schema={c: pl.Int64 for c in present_cols}).lazy()


# ── _sql_literal properties ─────────────────────────────────────────────────────


@given(st.integers(-(2**31), 2**31 - 1))
def test_sql_literal_int_contains_value(value: int) -> None:
    """Integer literals embed the numeric value without single quotes."""
    lit = _sql_literal(value, pl.Int64)
    assert str(value) in lit
    assert "'" not in lit


@given(st.integers(-(2**15), 2**15 - 1))
def test_sql_literal_int16_uses_smallint(value: int) -> None:
    """Int16 column literals use SMALLINT cast."""
    lit = _sql_literal(value, pl.Int16)
    assert "SMALLINT" in lit


@given(st.integers(0, 2**31 - 1))
def test_sql_literal_int32_uses_int(value: int) -> None:
    """Int32 column literals use INT cast."""
    lit = _sql_literal(value, pl.Int32)
    assert lit == f"CAST({value} AS INT)"


@given(st.text(max_size=100))
def test_sql_literal_string_is_quoted(value: str) -> None:
    """String literals are always wrapped in single quotes."""
    lit = _sql_literal(value, pl.String)
    assert lit.startswith("'")
    assert lit.endswith("'")


@given(
    st.builds(
        lambda prefix, suffix: prefix + "'" + suffix,
        prefix=st.text(max_size=50),
        suffix=st.text(max_size=50),
    )
)
def test_sql_literal_string_escapes_single_quotes(value: str) -> None:
    """Single quotes inside string values are escaped as ''."""
    lit = _sql_literal(value, pl.String)
    inner = lit[1:-1]
    # Every original single quote must have been doubled.
    assert inner.count("'") == value.count("'") * 2


@given(st.booleans())
def test_sql_literal_boolean(value: bool) -> None:
    """Boolean literals render as 'true' or 'false' (lowercase, no quotes)."""
    lit = _sql_literal(value, pl.Boolean)
    assert lit in ("true", "false")
    assert lit == ("true" if value else "false")


# ── schema: on_extra properties ─────────────────────────────────────────────────


@given(schema_and_frame_with_extras())
def test_schema_on_extra_drop_returns_only_expected_columns(spec) -> None:
    """on_extra='drop' always produces a frame with exactly the expected columns."""
    expected, lf = spec

    @schema(expect=expected, on_extra="drop")
    def identity(frame: pl.LazyFrame) -> pl.LazyFrame:
        return frame

    result = identity(lf).collect()
    assert set(result.columns) == set(expected.keys())


@given(schema_and_frame_with_extras())
def test_schema_on_extra_ignore_keeps_extra_columns(spec) -> None:
    """on_extra='ignore' passes all columns through unchanged."""
    expected, lf = spec
    original_cols = set(lf.collect_schema().names())

    @schema(expect=expected, on_extra="ignore")
    def identity(frame: pl.LazyFrame) -> pl.LazyFrame:
        return frame

    result = identity(lf).collect()
    assert set(result.columns) == original_cols


@given(schema_and_frame_with_extras())
def test_schema_on_extra_raise_raises_schema_error(spec) -> None:
    """on_extra='raise' raises SchemaError when extra columns are present."""
    expected, lf = spec

    @schema(expect=expected, on_extra="raise")
    def identity(frame: pl.LazyFrame) -> pl.LazyFrame:
        return frame

    with pytest.raises(SchemaError):
        identity(lf)


@given(matching_schema_and_frame())
def test_schema_matching_frame_always_passes_on_extra_raise(spec) -> None:
    """on_extra='raise' does NOT raise when the frame has exactly the expected columns."""
    expected, lf = spec

    @schema(expect=expected, on_extra="raise")
    def identity(frame: pl.LazyFrame) -> pl.LazyFrame:
        return frame

    result = identity(lf).collect()
    assert set(result.columns) == set(expected.keys())


# ── schema: on_missing properties ───────────────────────────────────────────────


@given(schema_and_frame_with_missing())
def test_schema_on_missing_raise_raises_schema_error(spec) -> None:
    """on_missing='raise' raises SchemaError when expected columns are absent."""
    expected, lf = spec

    @schema(expect=expected, on_missing="raise")
    def identity(frame: pl.LazyFrame) -> pl.LazyFrame:
        return frame

    with pytest.raises(SchemaError) as exc_info:
        identity(lf)
    # Violations should identify the missing columns.
    missing_cols = set(expected) - set(lf.collect_schema().names())
    violation_cols = {v.column for v in exc_info.value.violations}
    assert missing_cols <= violation_cols


@given(schema_and_frame_with_missing())
def test_schema_on_missing_drop_returns_empty_frame(spec) -> None:
    """on_missing='drop' returns an empty LazyFrame (no rows) when columns are absent."""
    expected, lf = spec

    @schema(expect=expected, on_missing="drop")
    def identity(frame: pl.LazyFrame) -> pl.LazyFrame:
        return frame

    result = identity(lf).collect()
    assert result.height == 0


@given(matching_schema_and_frame())
def test_schema_matching_frame_passes_on_missing_raise(spec) -> None:
    """on_missing='raise' does NOT raise when the frame has all expected columns."""
    expected, lf = spec

    @schema(expect=expected, on_missing="raise")
    def identity(frame: pl.LazyFrame) -> pl.LazyFrame:
        return frame

    result = identity(lf).collect()
    assert set(expected.keys()).issubset(set(result.columns))


# ── schema: evolution properties ────────────────────────────────────────────────


@given(matching_schema_and_frame())
def test_schema_evolution_cast_noop_on_matching_types(spec) -> None:
    """evolution='cast' on an already-correct schema is a no-op."""
    expected, lf = spec

    @schema(expect=expected, evolution="cast")
    def identity(frame: pl.LazyFrame) -> pl.LazyFrame:
        return frame

    result = identity(lf).collect()
    assert set(result.columns) == set(expected.keys())
    for col, dtype in expected.items():
        assert result[col].dtype == dtype


@given(matching_schema_and_frame())
def test_schema_row_count_preserved_on_valid_frame(spec) -> None:
    """A valid frame passes through with the same number of rows."""
    expected, lf = spec
    original_rows = lf.collect().height

    @schema(expect=expected)
    def identity(frame: pl.LazyFrame) -> pl.LazyFrame:
        return frame

    result = identity(lf).collect()
    assert result.height == original_rows


# ── _partition_predicate properties ─────────────────────────────────────────────


@given(
    st.lists(
        st.integers(0, 100),
        min_size=1,
        max_size=10,
        unique=True,
    )
)
def test_partition_predicate_int_contains_all_values(values: list[int]) -> None:
    """The predicate string embeds all distinct partition values."""
    df = pl.DataFrame({"part": values}, schema={"part": pl.Int64})
    pred = _partition_predicate(df, ["part"])
    assert pred is not None
    for v in values:
        assert str(v) in pred


@given(
    st.lists(
        st.text(alphabet=string.ascii_letters, min_size=1, max_size=8),
        min_size=1,
        max_size=5,
        unique=True,
    )
)
def test_partition_predicate_string_values_are_quoted(values: list[str]) -> None:
    """String partition values appear as single-quoted literals in the predicate."""
    df = pl.DataFrame({"part": values}, schema={"part": pl.String})
    pred = _partition_predicate(df, ["part"])
    assert pred is not None
    for v in values:
        assert f"'{v}'" in pred


@given(
    st.lists(st.integers(0, 50), min_size=1, max_size=10),
)
def test_partition_predicate_none_on_null_values(values: list[int]) -> None:
    """A null in the partition column causes _partition_predicate to return None."""
    data = values + [None]
    df = pl.DataFrame({"part": data}, schema={"part": pl.Int64})
    pred = _partition_predicate(df, ["part"])
    assert pred is None
