"""
schema.py

Validation layer that intercepts incoming LazyFrames before they reach the
transform function. Entirely decoupled from watermarks, Delta writes, and SCD
semantics — it only speaks LazyFrame.

The reason this is a separate module rather than logic inside IncrementalPipeline
is the decorator composition model. @schema wraps the transform function
directly; @incremental wraps the result. That means schema.py needs to be
importable and usable independently of incremental.py — a user should be able
to apply @schema to any LazyFrame -> LazyFrame function, not just ones managed
by this library.

Keeping validation here also means the failure modes (raise, drop, quarantine)
and their side effects (writing to a quarantine Delta table) are co-located.
There is one place to look when a batch is unexpectedly rejected.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from functools import wraps
from typing import Callable, Literal

import polars as pl
from deltalake import write_deltalake

logger = logging.getLogger(__name__)


class SchemaViolation:
    """A single schema violation detected during validation.

    Attributes:
        column:   Name of the column involved.
        issue:    ``"missing"``, ``"type_mismatch"``, or ``"extra"``.
        expected: The expected dtype as a string, or ``None`` for
                  ``"extra"`` issues.
        actual:   The actual dtype as a string, or ``None`` for
                  ``"missing"`` issues.

    Examples:
        ::

            v = SchemaViolation("price", "type_mismatch", "Float64", "String")
            print(v)           # SchemaViolation(column='price', issue='type_mismatch', ...)
            print(v.to_dict()) # {"column": "price", "issue": "type_mismatch", ...}
    """

    def __init__(
        self,
        column: str,
        issue: str,
        expected: str | None = None,
        actual: str | None = None,
    ) -> None:
        self.column = column
        self.issue = issue
        self.expected = expected
        self.actual = actual

    def to_dict(self) -> dict:
        return {
            "column": self.column,
            "issue": self.issue,
            "expected": self.expected,
            "actual": self.actual,
        }

    def __repr__(self) -> str:
        return (
            f"SchemaViolation(column={self.column!r}, issue={self.issue!r}, "
            f"expected={self.expected!r}, actual={self.actual!r})"
        )


class SchemaError(Exception):
    """Raised when schema validation fails and ``on_missing="raise"`` or
    ``evolution="strict"``.

    Attributes:
        violations: List of :class:`SchemaViolation` instances describing
                    every column-level issue found in the batch.

    Examples:
        ::

            try:
                result = transform(lf).collect()
            except SchemaError as e:
                print(e)  # human-readable summary of all violations
                for v in e.violations:
                    print(v.column, v.issue, v.expected, v.actual)
    """

    def __init__(self, violations: list[SchemaViolation]) -> None:
        """Initialise with a list of violations and format a human-readable
        message listing each column, issue, expected dtype, and actual dtype.
        """
        self.violations = violations
        lines: list[str] = []
        for v in violations:
            if v.issue == "missing":
                lines.append(f"  column {v.column!r}: missing (expected {v.expected})")
            elif v.issue == "type_mismatch":
                lines.append(f"  column {v.column!r}: expected {v.expected}, got {v.actual}")
            else:
                lines.append(f"  column {v.column!r}: unexpected extra column")
        super().__init__("Schema validation failed:\n" + "\n".join(lines))


def _write_quarantine(
    lf: pl.LazyFrame,
    violations: list[SchemaViolation],
    store: str | None,
) -> None:
    """Write a rejected batch to the quarantine Delta table.

    Collects *lf* eagerly (the only point in the library where a bad batch
    is materialised), appends a ``_violations`` column containing a JSON
    representation of *violations*, and appends a ``_quarantined_at``
    timestamp column. Writes to *store* with ``mode="append"`` so the
    quarantine table is a full history of all rejected batches.

    Does nothing silently if *store* is ``None``, allowing callers to use
    ``on_missing="quarantine"`` without configuring a quarantine path during
    development.

    Args:
        lf:         The rejected LazyFrame.
        violations: Violations that caused the rejection.
        store:      Local path to the quarantine Delta table, or ``None``.
    """
    if store is None:
        return
    try:
        _raw = lf.collect()
        assert isinstance(_raw, pl.DataFrame)
        df = _raw
        violations_json = json.dumps([v.to_dict() for v in violations])
        df = df.with_columns(
            pl.lit(violations_json).alias("_violations"),
            pl.lit(datetime.now(timezone.utc)).alias("_quarantined_at"),
        )
        write_deltalake(store, df.to_arrow(), mode="append")
    except Exception as exc:
        logger.warning("Failed to write quarantine batch: %s", exc)


def schema(
    expect: dict[str, pl.DataType],
    on_missing: Literal["raise", "drop", "quarantine"] = "raise",
    on_extra: Literal["ignore", "raise", "drop"] = "ignore",
    evolution: Literal["strict", "cast", "merge"] = "strict",
    quarantine: str | None = None,
) -> Callable:
    """Decorator that validates and normalises a LazyFrame's schema before
    it reaches the wrapped transform function.

    Operates on the schema metadata only (via ``lf.collect_schema()``) until
    a violation triggers quarantine, keeping the pipeline lazy throughout
    the happy path.

    Columns prefixed with ``_`` (``_source_file``, ``_ingested_at``) are
    injected by the library and are always exempt from ``on_extra`` checks.

    Args:
        expect:     Mapping of expected column names to Polars dtypes.
        on_missing: Action when an expected column is absent.
                    ``"raise"``      — raise :class:`SchemaError`.
                    ``"drop"``       — return an empty LazyFrame; skip the
                                       batch silently.
                    ``"quarantine"`` — write the batch to *quarantine* and
                                       return an empty LazyFrame.
        on_extra:   Action when unexpected columns are present.
                    ``"ignore"``     — pass them through unchanged.
                    ``"raise"``      — raise :class:`SchemaError`.
                    ``"drop"``       — remove them before the transform.
        evolution:  Action when a column is present but has the wrong dtype.
                    ``"strict"``     — raise :class:`SchemaError`.
                    ``"cast"``       — attempt a Polars cast to the expected
                                       type; raise if the cast is not safe.
                    ``"merge"``      — same as ``"cast"`` but only widens;
                                       mirrors Delta's ``schema_mode="merge"``
                                       semantics.
        quarantine: Local path to a Delta table for rejected batches. Only
                    used when ``on_missing="quarantine"``.

    Returns:
        A decorator that wraps a ``LazyFrame → LazyFrame`` function with
        schema validation. Composes correctly as the inner decorator under
        ``@incremental``.

    Raises:
        SchemaError: When validation fails and the relevant ``on_*`` mode is
                     ``"raise"`` or ``evolution="strict"``.

    Examples:
        Raise on any missing column (default)::

            @schema(expect={"id": pl.Int64, "value": pl.Float64})
            def transform(lf: pl.LazyFrame) -> pl.LazyFrame:
                return lf

        Quarantine bad batches, drop extra columns, cast type mismatches::

            @incremental(source="/data/uploads/", target="/data/delta/clean",
                         merge_on="id")
            @schema(
                expect={"id": pl.Int64, "value": pl.Float64},
                on_missing="quarantine",
                on_extra="drop",
                evolution="cast",
                quarantine="/data/delta/bad_batches",
            )
            def pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
                return lf.filter(pl.col("value") > 0)

        Standalone usage — validate without ``@incremental``::

            @schema(expect={"id": pl.Int64, "ts": pl.Datetime("us", "UTC")},
                    on_missing="raise", evolution="merge")
            def validate(lf: pl.LazyFrame) -> pl.LazyFrame:
                return lf

            clean_lf = validate(raw_lf)  # raises SchemaError if columns are missing

        Catch and inspect violations::

            try:
                validate(raw_lf).collect()
            except SchemaError as e:
                for v in e.violations:
                    print(v.column, v.issue, v.expected, v.actual)
    """

    def decorator(fn: Callable[[pl.LazyFrame], pl.LazyFrame]) -> Callable:
        @wraps(fn)
        def wrapper(lf: pl.LazyFrame) -> pl.LazyFrame:
            actual_schema = lf.collect_schema()
            violations: list[SchemaViolation] = []

            # Check expected columns for missing or type-mismatch.
            for col_name, expected_dtype in expect.items():
                if col_name not in actual_schema:
                    violations.append(SchemaViolation(col_name, "missing", str(expected_dtype)))
                else:
                    actual_dtype = actual_schema[col_name]
                    if actual_dtype != expected_dtype:
                        violations.append(
                            SchemaViolation(
                                col_name,
                                "type_mismatch",
                                str(expected_dtype),
                                str(actual_dtype),
                            )
                        )

            # Check for extra columns (library-injected "_*" columns are exempt).
            if on_extra != "ignore":
                for col_name in actual_schema:
                    if col_name.startswith("_"):
                        continue
                    if col_name not in expect:
                        violations.append(SchemaViolation(col_name, "extra"))

            missing_violations = [v for v in violations if v.issue == "missing"]
            type_violations = [v for v in violations if v.issue == "type_mismatch"]
            extra_violations = [v for v in violations if v.issue == "extra"]

            # Handle missing columns.
            if missing_violations:
                if on_missing == "raise":
                    raise SchemaError(missing_violations)
                # "drop" or "quarantine": skip the batch.
                if on_missing == "quarantine":
                    _write_quarantine(lf, missing_violations, quarantine)
                return pl.LazyFrame()

            # Handle extra columns.
            if extra_violations:
                if on_extra == "raise":
                    raise SchemaError(extra_violations)
                if on_extra == "drop":
                    extra_cols = [v.column for v in extra_violations]
                    lf = lf.drop(extra_cols)

            # Handle type evolution.
            if type_violations:
                if evolution == "strict":
                    raise SchemaError(type_violations)
                # "cast" or "merge": attempt a Polars cast.
                for v in type_violations:
                    lf = lf.with_columns(pl.col(v.column).cast(expect[v.column]))

            return fn(lf)

        return wrapper

    return decorator
