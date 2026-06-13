"""Tests for schema validation module."""

import polars as pl
import pytest
from data_warehousing_with_polars.schema import SchemaError, SchemaViolation, schema

# ── SchemaViolation ───────────────────────────────────────────────────────────


def test_schema_violation_to_dict_missing():
    v = SchemaViolation("col", "missing", expected="Int64")
    assert v.to_dict() == {"column": "col", "issue": "missing", "expected": "Int64", "actual": None}


def test_schema_violation_to_dict_type_mismatch():
    v = SchemaViolation("col", "type_mismatch", expected="Int64", actual="Utf8")
    assert v.to_dict() == {
        "column": "col",
        "issue": "type_mismatch",
        "expected": "Int64",
        "actual": "Utf8",
    }


def test_schema_violation_to_dict_extra():
    v = SchemaViolation("col", "extra")
    assert v.to_dict() == {"column": "col", "issue": "extra", "expected": None, "actual": None}


def test_schema_violation_repr():
    v = SchemaViolation("col", "missing", "Int64")
    assert "col" in repr(v)
    assert "missing" in repr(v)


# ── SchemaError ───────────────────────────────────────────────────────────────


def test_schema_error_stores_violations():
    v = SchemaViolation("id", "missing", "Int64")
    err = SchemaError([v])
    assert err.violations == [v]


def test_schema_error_message_missing():
    v = SchemaViolation("id", "missing", "Int64")
    err = SchemaError([v])
    assert "id" in str(err)
    assert "Int64" in str(err)


def test_schema_error_message_type_mismatch():
    v = SchemaViolation("value", "type_mismatch", "Float64", "Int32")
    err = SchemaError([v])
    assert "value" in str(err)
    assert "Float64" in str(err)
    assert "Int32" in str(err)


# ── schema() decorator — happy path ──────────────────────────────────────────


def test_schema_happy_path_passes_through():
    @schema(expect={"id": pl.Int64, "value": pl.Float64})
    def transform(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    lf = pl.DataFrame({"id": [1, 2], "value": [1.0, 2.0]}).lazy()
    result = transform(lf).collect()
    assert len(result) == 2
    assert list(result["id"]) == [1, 2]


def test_schema_transform_function_applied():
    @schema(expect={"id": pl.Int64, "value": pl.Float64})
    def transform(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.filter(pl.col("value") > 1.5)

    lf = pl.DataFrame({"id": [1, 2, 3], "value": [1.0, 2.0, 3.0]}).lazy()
    result = transform(lf).collect()
    assert len(result) == 2


# ── on_missing ───────────────────────────────────────────────────────────────


def test_schema_missing_raise():
    @schema(expect={"id": pl.Int64, "value": pl.Float64}, on_missing="raise")
    def transform(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    lf = pl.DataFrame({"id": [1]}).lazy()
    with pytest.raises(SchemaError) as exc_info:
        transform(lf)
    assert any(v.issue == "missing" for v in exc_info.value.violations)


def test_schema_missing_drop_returns_empty():
    @schema(expect={"id": pl.Int64, "value": pl.Float64}, on_missing="drop")
    def transform(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    lf = pl.DataFrame({"id": [1]}).lazy()
    result = transform(lf).collect()
    assert result.is_empty()


def test_schema_missing_quarantine_no_store_returns_empty():
    @schema(
        expect={"id": pl.Int64, "value": pl.Float64},
        on_missing="quarantine",
        quarantine=None,
    )
    def transform(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    lf = pl.DataFrame({"id": [1]}).lazy()
    result = transform(lf).collect()
    assert result.is_empty()


def test_schema_missing_quarantine_writes_to_store(tmp_path):
    quarantine_path = str(tmp_path / "quarantine")

    @schema(
        expect={"id": pl.Int64, "value": pl.Float64},
        on_missing="quarantine",
        quarantine=quarantine_path,
    )
    def transform(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    lf = pl.DataFrame({"id": [1]}).lazy()
    transform(lf)

    q = pl.scan_delta(quarantine_path).collect()
    assert len(q) == 1
    assert "_violations" in q.columns
    assert "_quarantined_at" in q.columns


# ── on_extra ─────────────────────────────────────────────────────────────────


def test_schema_extra_ignore_passes_through():
    @schema(expect={"id": pl.Int64}, on_extra="ignore")
    def transform(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    lf = pl.DataFrame({"id": [1], "extra": ["x"]}).lazy()
    result = transform(lf).collect()
    assert "extra" in result.columns


def test_schema_extra_raise():
    @schema(expect={"id": pl.Int64}, on_extra="raise")
    def transform(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    lf = pl.DataFrame({"id": [1], "extra": ["x"]}).lazy()
    with pytest.raises(SchemaError) as exc_info:
        transform(lf)
    assert any(v.issue == "extra" for v in exc_info.value.violations)


def test_schema_extra_drop():
    @schema(expect={"id": pl.Int64}, on_extra="drop")
    def transform(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    lf = pl.DataFrame({"id": [1], "extra": ["x"]}).lazy()
    result = transform(lf).collect()
    assert "extra" not in result.columns
    assert "id" in result.columns


def test_schema_internal_columns_exempt_from_extra():
    """Columns starting with _ must not trigger on_extra="raise"."""

    @schema(expect={"id": pl.Int64}, on_extra="raise")
    def transform(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    lf = pl.DataFrame({"id": [1], "_source_file": ["/path"], "_ingested_at": [1]}).lazy()
    result = transform(lf).collect()
    assert "_source_file" in result.columns


# ── evolution ────────────────────────────────────────────────────────────────


def test_schema_evolution_strict_raises_on_mismatch():
    @schema(expect={"id": pl.Int64}, evolution="strict")
    def transform(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    lf = pl.DataFrame({"id": [1, 2]}).lazy().with_columns(pl.col("id").cast(pl.Int32))
    with pytest.raises(SchemaError) as exc_info:
        transform(lf)
    assert any(v.issue == "type_mismatch" for v in exc_info.value.violations)


def test_schema_evolution_cast_coerces_type():
    @schema(expect={"id": pl.Int64}, evolution="cast")
    def transform(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    lf = pl.DataFrame({"id": [1, 2]}).lazy().with_columns(pl.col("id").cast(pl.Int32))
    result = transform(lf).collect()
    assert result["id"].dtype == pl.Int64


def test_schema_evolution_merge_coerces_type():
    @schema(expect={"id": pl.Int64}, evolution="merge")
    def transform(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    lf = pl.DataFrame({"id": [1, 2]}).lazy().with_columns(pl.col("id").cast(pl.Int32))
    result = transform(lf).collect()
    assert result["id"].dtype == pl.Int64
