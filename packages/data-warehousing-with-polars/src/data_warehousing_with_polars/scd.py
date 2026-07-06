"""
scd.py

SCD Type 2 (valid_from/valid_to history) and Type 4 (separate history table) write semantics.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from datetime import date, datetime, timezone
from typing import cast

import polars as pl
from deltalake import CommitProperties, DeltaTable, WriterProperties, write_deltalake
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


def _align_join_dtypes(
    lf: pl.LazyFrame, existing_schema: pl.Schema, cols: list[str]
) -> pl.LazyFrame:
    """Cast *lf*'s *cols* to match *existing_schema* wherever they differ.

    A batch computed fresh by the caller's own code can end up in a different
    dtype than what's actually persisted for the same logical column — most
    commonly ``Datetime`` precision: Delta normalises stored timestamps to
    microseconds on write regardless of what precision was written, but a
    freshly recomputed batch keeps whatever precision the caller's own code
    produces (e.g. milliseconds, matching an upstream API's native units).
    Joining directly on such a column fails with a polars ``SchemaError``,
    since join keys require an exact dtype match. Casting the incoming side
    to the already-persisted dtype is safe — it doesn't change what's stored,
    only reconciles this comparison to what's already on disk.
    """
    lf_schema = lf.collect_schema()
    casts = [
        pl.col(c).cast(existing_schema[c])
        for c in cols
        if c in existing_schema and c in lf_schema and existing_schema[c] != lf_schema[c]
    ]
    return lf.with_columns(casts) if casts else lf


def _upsert_overwrite(
    target: str,
    df: pl.DataFrame,
    join_cols: list[str],
    partition_list: list[str] | None,
    commit_properties: CommitProperties | None = None,
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
        new_lf = _align_join_dtypes(df.lazy(), existing.collect_schema(), join_cols)
        keep = existing.join(new_lf.select(join_cols), on=join_cols, how="anti")
        merged = pl.concat([keep, new_lf], how="diagonal_relaxed")
        merged.sink_delta(
            target,
            mode="overwrite",
            delta_write_options={
                "predicate": predicate,
                "partition_by": partition_list,
                "writer_properties": WriterProperties(),
                "commit_properties": commit_properties,
            },
        )
        return

    existing = pl.scan_delta(target)
    new_lf = _align_join_dtypes(df.lazy(), existing.collect_schema(), join_cols)
    keep = existing.join(new_lf.select(join_cols), on=join_cols, how="anti")
    merged = pl.concat([keep, new_lf], how="diagonal_relaxed")
    merged.sink_delta(
        target,
        mode="overwrite",
        delta_write_options={
            "schema_mode": "overwrite",
            "configuration": _CDF_CONFIG,
            "partition_by": partition_list,
            "writer_properties": WriterProperties(),
            "commit_properties": commit_properties,
        },
    )


def _sink_scd2(
    target: str,
    lf: pl.LazyFrame,
    merge_on: str | list[str],
    partition_by: str | list[str] | None = None,
    commit_properties: CommitProperties | None = None,
    creation_lock: threading.Lock | None = None,
) -> None:
    """Write *lf* to a SCD Type 2 table, closing old versions and appending new ones.

    Injects ``valid_from``, ``valid_to``, and ``is_current`` columns. On subsequent
    runs: closes matching current rows (sets ``valid_to = now``, ``is_current = false``)
    and appends new (deduplicated on ``(merge_on, valid_from)``) rows in a single
    commit — one ``sink_delta`` overwrite of just the touched partitions (or the
    whole table, unpartitioned) — rather than a separate close-commit followed by a
    separate append-commit. Each extra commit costs more the larger the table's
    history gets (delta-rs reloads/validates the growing transaction log per
    commit), so halving the commit count here matters more as the table grows, not
    less — this is what makes ``by_partition=True`` a net win here instead of a net
    loss (two commits per partition, each against a table whose history is now N
    times longer, cost more than one bigger commit against a table touched once).
    """
    now = datetime.now(timezone.utc)
    keys = [merge_on] if isinstance(merge_on, str) else list(merge_on)
    partition_list = (
        [partition_by]
        if isinstance(partition_by, str)
        else (list(partition_by) if partition_by else None)
    )

    lf_versioned = lf.with_columns(
        pl.lit(now).alias("valid_from"),
        pl.lit(None).cast(pl.Datetime("us", "UTC")).alias("valid_to"),
        pl.lit(True).alias("is_current"),
    )

    with creation_lock or contextlib.nullcontext():
        try:
            DeltaTable(target)  # existence probe; raises TableNotFoundError on first run
            table_exists = True
        except TableNotFoundError:
            table_exists = False
            # Streaming create: data flows chunk-by-chunk without ever materialising
            # the whole first batch as one in-memory DataFrame (unlike the
            # collect() + write_deltalake() this replaces).
            lf_versioned.sink_delta(
                target,
                mode="overwrite",
                delta_write_options={
                    "configuration": _CDF_CONFIG,
                    "partition_by": partition_list,
                    "writer_properties": WriterProperties(),
                    "commit_properties": commit_properties,
                },
            )

    if not table_exists:
        return

    df = cast(pl.DataFrame, lf_versioned.collect(engine="streaming"))

    # Close existing current versions for keys in the incoming batch. Done as a
    # full-table rewrite rather than a Delta MERGE: delta-rs 1.6's MERGE is
    # unreliable on object-store (S3) targets at scale (see _upsert_overwrite). Only
    # rows that are currently open AND whose key appears in the batch are closed.
    #
    # When the target is partitioned (and the partition values render into a
    # ``replaceWhere`` predicate), the rewrite is narrowed to *just the partitions
    # the batch touches* — reading and overwriting only those partitions instead of
    # the whole table, exactly as ``_upsert_overwrite`` does. This assumes a key's
    # partition value is stable across runs (the same assumption upserts make).
    incoming = df.select(keys).unique().with_columns(pl.lit(True).alias("__match"))
    to_close = pl.col("__match").fill_null(False) & pl.col("is_current")
    predicate = _partition_predicate(df, partition_list) if partition_list else None

    # Both incoming's `keys` and lf_versioned's `dedup_cols` are computed fresh by
    # this run and may not match the target's actual persisted dtypes (e.g.
    # Datetime precision — see _align_join_dtypes) — align both to the target's
    # schema once, up front, so every join below compares like-for-like.
    target_schema = pl.scan_delta(target).collect_schema()
    incoming_lf = _align_join_dtypes(incoming.lazy(), target_schema, keys)

    def _close(existing: pl.LazyFrame) -> pl.LazyFrame:
        return (
            existing.join(incoming_lf, on=keys, how="left")
            .with_columns(
                pl.when(to_close).then(pl.lit(now)).otherwise(pl.col("valid_to")).alias("valid_to"),
                pl.when(to_close)
                .then(pl.lit(False))
                .otherwise(pl.col("is_current"))
                .alias("is_current"),
            )
            .drop("__match")
        )

    # New rows are deduplicated on (key, valid_from) for idempotency when the
    # pipeline is re-run on the same batch. `_close` never changes `valid_from`, so
    # `existing`'s (key, valid_from) set is the same whether read before or after
    # this commit — one scan of `existing` safely covers both the close-join and
    # this anti-join, rather than a second, separate scan later.
    dedup_cols = keys + ["valid_from"]
    lf_versioned_aligned = _align_join_dtypes(lf_versioned, target_schema, dedup_cols)

    if predicate is not None:
        assert partition_list is not None
        existing = pl.scan_delta(target)
        for col in partition_list:
            existing = existing.filter(pl.col(col).is_in(df[col].unique().to_list()))
        new_rows = lf_versioned_aligned.join(existing.select(dedup_cols), on=dedup_cols, how="anti")
        pl.concat([_close(existing), new_rows], how="diagonal_relaxed").sink_delta(
            target,
            mode="overwrite",
            delta_write_options={
                "predicate": predicate,
                "partition_by": partition_list,
                "writer_properties": WriterProperties(),
                "commit_properties": commit_properties,
            },
        )
    else:
        existing = pl.scan_delta(target)
        new_rows = lf_versioned_aligned.join(existing.select(dedup_cols), on=dedup_cols, how="anti")
        pl.concat([_close(existing), new_rows], how="diagonal_relaxed").sink_delta(
            target,
            mode="overwrite",
            delta_write_options={
                "schema_mode": "overwrite",
                "configuration": _CDF_CONFIG,
                "partition_by": partition_list,
                "writer_properties": WriterProperties(),
                "commit_properties": commit_properties,
            },
        )


def _sink_scd4(
    target: str,
    history_target: str,
    lf: pl.LazyFrame,
    merge_on: str | list[str],
    partition_by: str | list[str] | None = None,
    commit_properties: CommitProperties | None = None,
    creation_lock: threading.Lock | None = None,
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

    lf_dedup = lf.unique(subset=keys, keep="last")

    with creation_lock or contextlib.nullcontext():
        try:
            DeltaTable(target)  # existence probe; raises TableNotFoundError on first run
            target_exists = True
        except TableNotFoundError:
            target_exists = False
            # Streaming create: data flows chunk-by-chunk without ever materialising
            # the whole first batch as one in-memory DataFrame (unlike the
            # collect() + write_deltalake() this replaces).
            lf_dedup.sink_delta(
                target,
                mode="overwrite",
                delta_write_options={
                    "configuration": _CDF_CONFIG,
                    "partition_by": partition_list,
                    "writer_properties": WriterProperties(),
                    "commit_properties": commit_properties,
                },
            )

    if not target_exists:
        return

    df = cast(pl.DataFrame, lf_dedup.collect(engine="streaming"))

    # Step 1: Archive current versions of affected records. The target is scanned
    # out-of-core via the streaming engine; the inner join against the batch's
    # distinct keys bounds the result to one row per incoming key.
    existing_for_archive = pl.scan_delta(target)
    incoming_keys = _align_join_dtypes(
        df.select(keys).unique().lazy(), existing_for_archive.collect_schema(), keys
    )
    _current = (
        existing_for_archive.join(incoming_keys, on=keys, how="inner")
        .with_columns(pl.lit(now).alias("superseded_at"))
        .collect(engine="streaming")
    )
    current = cast(pl.DataFrame, _current)
    if len(current) > 0:
        with creation_lock or contextlib.nullcontext():
            try:
                DeltaTable(history_target)  # existence probe
                history_exists = True
            except TableNotFoundError:
                write_deltalake(
                    history_target,
                    current.to_arrow(),
                    mode="overwrite",
                    configuration=_CDF_CONFIG,
                    writer_properties=WriterProperties(),
                    commit_properties=commit_properties,
                )
                history_exists = False
        if history_exists:
            write_deltalake(
                history_target,
                current.to_arrow(),
                mode="append",
                writer_properties=WriterProperties(),
                commit_properties=commit_properties,
            )

    # Step 2: Upsert current state (SCD Type 1 semantics).
    _upsert_overwrite(target, df, keys, partition_list, commit_properties=commit_properties)
