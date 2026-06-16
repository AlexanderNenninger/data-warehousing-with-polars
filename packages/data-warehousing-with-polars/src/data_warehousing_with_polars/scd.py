"""
scd.py

SCD Type 2 (valid_from/valid_to history) and Type 4 (separate history table) write semantics.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import cast

import polars as pl
from deltalake import DeltaTable, WriterProperties, write_deltalake
from deltalake.exceptions import TableNotFoundError

logger = logging.getLogger(__name__)

_CDF_CONFIG = {"delta.enableChangeDataFeed": "true"}


# Maps polars integer dtypes to SQL type names so partition literals match the
# column's exact width. delta-rs's kernel refuses mixed-width comparisons (e.g.
# Int32 column vs an unqualified — Int64 — literal), so integers must be cast.
_INT_SQL_TYPE = {
    "Int8": "TINYINT",
    "Int16": "SMALLINT",
    "Int32": "INT",
    "Int64": "BIGINT",
    "UInt8": "SMALLINT",
    "UInt16": "INT",
    "UInt32": "BIGINT",
    "UInt64": "BIGINT",
}


def _sql_literal(value: object, dtype: pl.DataType) -> str:
    """Render *value* as a SQL literal for a Delta ``replaceWhere`` predicate.

    Integers and dates are wrapped in a ``CAST`` to the column's exact type so the
    kernel does not reject a width mismatch when evaluating the predicate.
    """
    if dtype == pl.Boolean:
        return "true" if value else "false"
    if dtype.is_integer():
        return f"CAST({value} AS {_INT_SQL_TYPE.get(str(dtype), 'BIGINT')})"
    if dtype == pl.Date:
        iso = value.isoformat() if isinstance(value, date) else str(value)
        return f"CAST('{iso}' AS DATE)"
    return "'" + str(value).replace("'", "''") + "'"


def _partition_predicate(df: pl.DataFrame, partition_cols: list[str]) -> str | None:
    """Build a ``replaceWhere`` predicate covering every partition *df* touches.

    Returns ``col IN (...) AND col2 IN (...)`` over the distinct partition values in
    *df*, or ``None`` if any partition column has a null or a dtype we don't render
    safely — in which case the caller falls back to a full-table rewrite.
    """
    schema = df.schema
    clauses: list[str] = []
    for col in partition_cols:
        dtype = schema[col]
        if df[col].null_count() > 0:
            return None
        if not (dtype.is_integer() or dtype in (pl.Utf8, pl.Boolean, pl.Date)):
            return None
        values = df[col].unique().sort().to_list()
        literals = ", ".join(_sql_literal(v, dtype) for v in values)
        clauses.append(f"{col} IN ({literals})")
    return " AND ".join(clauses)


def _upsert_overwrite(
    target: str,
    df: pl.DataFrame,
    join_cols: list[str],
    partition_list: list[str] | None,
) -> None:
    """Upsert *df* into *target* by rewriting only the affected data, avoiding ``MERGE``.

    delta-rs 1.6's ``MERGE`` is unusable here: it fails deterministically with
    "matched a target row with multiple source rows" once a single merge matches
    more than 8192 rows, even though the join keys are provably unique (see the
    repro in the project notes). So we never issue a ``MERGE``.

    **Partitioned target** (``partition_list`` set and renderable): read only the
    partitions the batch touches, drop rows whose *join_cols* are in the batch
    (anti-join), concatenate the new rows, and ``replaceWhere``-overwrite *just those
    partitions* in one atomic commit. Cost scales with the data the batch touches,
    not the table — so arbitrarily large tables are fine as long as a batch lands in
    a bounded set of partitions.

    **Unpartitioned target** (or a partition dtype we can't render into a predicate):
    fall back to a full-table anti-join + overwrite. Without partitioning there is no
    way to localise the write; partition the target to scale.
    """
    predicate = _partition_predicate(df, partition_list) if partition_list else None

    if predicate is not None:
        assert partition_list is not None
        existing = pl.scan_delta(target)
        for col in partition_list:
            existing = existing.filter(pl.col(col).is_in(df[col].unique().to_list()))
        keep = existing.join(df.lazy().select(join_cols), on=join_cols, how="anti")
        merged = pl.concat([keep, df.lazy()], how="diagonal_relaxed")
        merged.sink_delta(
            target,
            mode="overwrite",
            delta_write_options={
                "predicate": predicate,
                "partition_by": partition_list,
                "writer_properties": WriterProperties(),
            },
        )
        return

    keep = pl.scan_delta(target).join(df.lazy().select(join_cols), on=join_cols, how="anti")
    merged = pl.concat([keep, df.lazy()], how="diagonal_relaxed")
    merged.sink_delta(
        target,
        mode="overwrite",
        delta_write_options={
            "schema_mode": "overwrite",
            "configuration": _CDF_CONFIG,
            "partition_by": partition_list,
            "writer_properties": WriterProperties(),
        },
    )


def _sink_scd2(
    target: str,
    lf: pl.LazyFrame,
    merge_on: str | list[str],
    partition_by: str | list[str] | None = None,
) -> None:
    """Write *lf* to a SCD Type 2 table, closing old versions and appending new ones.

    Injects ``valid_from``, ``valid_to``, and ``is_current`` columns. On subsequent
    runs: closes matching current rows (sets ``valid_to = now``, ``is_current = false``),
    then appends new rows. Deduplicates on ``(merge_on, valid_from)`` for idempotency.
    """
    now = datetime.now(timezone.utc)
    keys = [merge_on] if isinstance(merge_on, str) else list(merge_on)
    partition_list = (
        [partition_by]
        if isinstance(partition_by, str)
        else (list(partition_by) if partition_by else None)
    )

    _raw = lf.with_columns(
        pl.lit(now).alias("valid_from"),
        pl.lit(None).cast(pl.Datetime("us", "UTC")).alias("valid_to"),
        pl.lit(True).alias("is_current"),
    ).collect()
    df = cast(pl.DataFrame, _raw)

    try:
        DeltaTable(target)  # existence probe; raises TableNotFoundError on first run
    except TableNotFoundError:
        write_deltalake(
            target,
            df.to_arrow(),
            mode="overwrite",
            configuration=_CDF_CONFIG,
            partition_by=partition_list,
            writer_properties=WriterProperties(),
        )
        return

    # Step 1: Close existing current versions for keys in the incoming batch.
    # Done as a full-table rewrite rather than a Delta MERGE: delta-rs 1.6's MERGE is
    # unreliable on object-store (S3) targets at scale (see _upsert_overwrite). Only
    # rows that are currently open AND whose key appears in the batch are closed.
    incoming = df.select(keys).unique().with_columns(pl.lit(True).alias("__match"))
    to_close = pl.col("__match").fill_null(False) & pl.col("is_current")
    closed = (
        pl
        .scan_delta(target)
        .join(incoming.lazy(), on=keys, how="left")
        .with_columns(
            pl.when(to_close).then(pl.lit(now)).otherwise(pl.col("valid_to")).alias("valid_to"),
            pl
            .when(to_close)
            .then(pl.lit(False))
            .otherwise(pl.col("is_current"))
            .alias("is_current"),
        )
        .drop("__match")
    )
    closed.sink_delta(
        target,
        mode="overwrite",
        delta_write_options={
            "schema_mode": "overwrite",
            "configuration": _CDF_CONFIG,
            "partition_by": partition_list,
            "writer_properties": WriterProperties(),
        },
    )

    # Step 2: Append new current versions, deduplicating on (key, valid_from)
    # for idempotency when the pipeline is re-run on the same batch. The existing
    # ``(key, valid_from)`` set is scanned out-of-core via the streaming engine; the
    # anti-join result is bounded by the incoming batch, so it materialises safely.
    dedup_cols = keys + ["valid_from"]
    new_df = cast(
        pl.DataFrame,
        df
        .lazy()
        .join(pl.scan_delta(target).select(dedup_cols), on=dedup_cols, how="anti")
        .collect(engine="streaming"),
    )
    if len(new_df) > 0:
        write_deltalake(
            target, new_df.to_arrow(), mode="append", writer_properties=WriterProperties()
        )


def _sink_scd4(
    target: str,
    history_target: str,
    lf: pl.LazyFrame,
    merge_on: str | list[str],
    partition_by: str | list[str] | None = None,
) -> None:
    """Write *lf* to a SCD Type 4 table pair.

    Archives superseded rows to *history_target* (with ``superseded_at``),
    then upserts current state into *target*.
    """
    now = datetime.now(timezone.utc)
    keys = [merge_on] if isinstance(merge_on, str) else list(merge_on)
    partition_list = (
        [partition_by]
        if isinstance(partition_by, str)
        else (list(partition_by) if partition_by else None)
    )

    _raw = lf.collect()
    df = cast(pl.DataFrame, _raw).unique(subset=keys, keep="last")

    try:
        DeltaTable(target)  # existence probe; raises TableNotFoundError on first run

        # Step 1: Archive current versions of affected records. The target is scanned
        # out-of-core via the streaming engine; the inner join against the batch's
        # distinct keys bounds the result to one row per incoming key.
        incoming_keys = df.select(keys).unique()
        _current = (
            pl
            .scan_delta(target)
            .join(incoming_keys.lazy(), on=keys, how="inner")
            .with_columns(pl.lit(now).alias("superseded_at"))
            .collect(engine="streaming")
        )
        current = cast(pl.DataFrame, _current)
        if len(current) > 0:
            try:
                write_deltalake(
                    history_target,
                    current.to_arrow(),
                    mode="append",
                    writer_properties=WriterProperties(),
                )
            except TableNotFoundError:
                write_deltalake(
                    history_target,
                    current.to_arrow(),
                    mode="overwrite",
                    configuration=_CDF_CONFIG,
                    writer_properties=WriterProperties(),
                )

        # Step 2: Upsert current state (SCD Type 1 semantics).
        _upsert_overwrite(target, df, keys, partition_list)

    except TableNotFoundError:
        write_deltalake(
            target,
            df.to_arrow(),
            mode="overwrite",
            configuration=_CDF_CONFIG,
            partition_by=partition_list,
            writer_properties=WriterProperties(),
        )
