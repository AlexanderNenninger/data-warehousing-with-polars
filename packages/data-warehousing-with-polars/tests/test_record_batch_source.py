"""Tests for make_lazy: PyArrow RecordBatchReader → Polars LazyFrame adapter."""

import polars as pl
import pyarrow as pa

from data_warehousing_with_polars.record_batch_source import make_lazy


def _make_reader(
    fields: list[tuple[str, pa.DataType]],
    batches: list[dict],
) -> pa.RecordBatchReader:
    schema = pa.schema([pa.field(name, dtype) for name, dtype in fields])
    arrow_batches = [pa.record_batch(data, schema=schema) for data in batches]
    return pa.RecordBatchReader.from_batches(schema, arrow_batches)


# ── Schema inference ──────────────────────────────────────────────────────────


def test_make_lazy_returns_lazy_frame():
    reader = _make_reader([("id", pa.int64())], [{"id": [1, 2]}])
    lf = make_lazy(reader)
    assert isinstance(lf, pl.LazyFrame)


def test_make_lazy_schema_matches_arrow_schema():
    reader = _make_reader(
        [("id", pa.int64()), ("name", pa.string())],
        [{"id": [1], "name": ["Alice"]}],
    )
    lf = make_lazy(reader)
    schema = lf.collect_schema()
    assert "id" in schema
    assert "name" in schema


def test_make_lazy_schema_from_empty_reader():
    """Schema must be derivable even when no batches are present."""
    schema = pa.schema([pa.field("id", pa.int64()), pa.field("value", pa.float64())])
    reader = pa.RecordBatchReader.from_batches(schema, [])
    lf = make_lazy(reader)
    assert "id" in lf.collect_schema()
    assert "value" in lf.collect_schema()


# ── Data yielding ─────────────────────────────────────────────────────────────


def test_make_lazy_single_batch():
    reader = _make_reader(
        [("id", pa.int64()), ("val", pa.float64())],
        [{"id": [1, 2, 3], "val": [1.0, 2.0, 3.0]}],
    )
    df = make_lazy(reader).collect()
    assert isinstance(df, pl.DataFrame), "type should be DataFrame after collect()"
    assert len(df) == 3
    assert list(df["id"]) == [1, 2, 3]


def test_make_lazy_multiple_batches_concatenated():
    reader = _make_reader(
        [("id", pa.int64())],
        [{"id": [1, 2]}, {"id": [3, 4]}, {"id": [5]}],
    )
    df = make_lazy(reader).collect()
    assert isinstance(df, pl.DataFrame), "type should be DataFrame after collect()"
    assert len(df) == 5
    assert list(df["id"]) == [1, 2, 3, 4, 5]


def test_make_lazy_empty_reader_returns_empty_frame():
    schema = pa.schema([pa.field("id", pa.int64())])
    reader = pa.RecordBatchReader.from_batches(schema, [])
    df = make_lazy(reader).collect()
    assert isinstance(df, pl.DataFrame), "type should be DataFrame after collect()"
    assert df.is_empty()
    assert "id" in df.columns


# ── Column projection ─────────────────────────────────────────────────────────


def test_make_lazy_select_subset_of_columns():
    reader = _make_reader(
        [("id", pa.int64()), ("name", pa.string()), ("extra", pa.float64())],
        [{"id": [1], "name": ["Alice"], "extra": [0.5]}],
    )
    df = make_lazy(reader).select(["id", "name"]).collect()
    assert isinstance(df, pl.DataFrame), "type should be DataFrame after collect()"
    assert "id" in df.columns
    assert "name" in df.columns
    assert "extra" not in df.columns


# ── Row limiting ──────────────────────────────────────────────────────────────


def test_make_lazy_head_limits_rows():
    reader = _make_reader(
        [("id", pa.int64())],
        [{"id": list(range(50))}],
    )
    df = make_lazy(reader).head(10).collect()
    assert isinstance(df, pl.DataFrame), "type should be DataFrame after collect()"
    assert len(df) == 10


def test_make_lazy_head_across_batch_boundary():
    """head(n) spanning multiple batches must return exactly n rows."""
    reader = _make_reader(
        [("id", pa.int64())],
        [{"id": [1, 2, 3]}, {"id": [4, 5, 6]}, {"id": [7, 8, 9]}],
    )
    df = make_lazy(reader).head(5).collect()
    assert isinstance(df, pl.DataFrame), "type should be DataFrame after collect()"
    assert len(df) == 5


# ── Data types ────────────────────────────────────────────────────────────────


def test_make_lazy_int32_column():
    reader = _make_reader([("x", pa.int32())], [{"x": [1, 2, 3]}])
    df = make_lazy(reader).collect()
    assert isinstance(df, pl.DataFrame), "type should be DataFrame after collect()"
    assert df["x"].dtype == pl.Int32


def test_make_lazy_string_column():
    reader = _make_reader([("s", pa.string())], [{"s": ["a", "b"]}])
    df = make_lazy(reader).collect()
    assert isinstance(df, pl.DataFrame), "type should be DataFrame after collect()"
    assert df["s"].dtype in (pl.String, pl.Utf8)


def test_make_lazy_boolean_column():
    reader = _make_reader([("flag", pa.bool_())], [{"flag": [True, False]}])
    df = make_lazy(reader).collect()
    assert isinstance(df, pl.DataFrame), "type should be DataFrame after collect()"
    assert df["flag"].dtype == pl.Boolean
    assert list(df["flag"]) == [True, False]
