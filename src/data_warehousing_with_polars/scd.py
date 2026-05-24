"""
scd.py

SCD Type 2 (valid_from/valid_to history) and Type 4 (separate history table) write semantics.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import polars as pl
from deltalake import DeltaTable, WriterProperties, write_deltalake
from deltalake.exceptions import TableNotFoundError

logger = logging.getLogger(__name__)

_CDF_CONFIG = {"delta.enableChangeDataFeed": "true"}


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
    assert isinstance(_raw, pl.DataFrame)
    df = _raw

    try:
        dt = DeltaTable(target)
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
    # NOTE: We intentionally keep `is_current = true` as a *per-action* predicate
    # rather than in the outer merge predicate. The outer predicate triggers the
    # Delta kernel's file-skipping stats lookup, and boolean columns are absent from
    # minValues/maxValues in the stats JSON, causing a kernel error in deltalake 1.6.
    key_pred = " AND ".join(f"target.{k} = source.{k}" for k in keys)
    now_sql = now.strftime("%Y-%m-%dT%H:%M:%S.%f")
    (
        dt.merge(df.to_arrow(), predicate=key_pred, source_alias="source", target_alias="target")
        .when_matched_update(
            predicate="target.is_current = true",
            updates={
                "valid_to": f"CAST('{now_sql}' AS TIMESTAMP)",
                "is_current": "false",
            },
        )
        .execute()
    )

    # Step 2: Append new current versions, deduplicating on (key, valid_from)
    # for idempotency when the pipeline is re-run on the same batch.
    dedup_cols = keys + ["valid_from"]
    _existing = pl.scan_delta(target).select(dedup_cols).collect()
    assert isinstance(_existing, pl.DataFrame)
    new_df = df.join(_existing, on=dedup_cols, how="anti")
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
    assert isinstance(_raw, pl.DataFrame)
    df = _raw

    try:
        dt = DeltaTable(target)

        # Step 1: Archive current versions of affected records.
        incoming_keys = df.select(keys).unique()
        _current = (
            pl.scan_delta(target)
            .join(incoming_keys.lazy(), on=keys, how="inner")
            .with_columns(pl.lit(now).alias("superseded_at"))
            .collect()
        )
        assert isinstance(_current, pl.DataFrame)
        current = _current
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
        key_pred = " AND ".join(f"target.{k} = source.{k}" for k in keys)
        (
            dt.merge(
                df.to_arrow(), predicate=key_pred, source_alias="source", target_alias="target"
            )
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute()
        )

    except TableNotFoundError:
        write_deltalake(
            target,
            df.to_arrow(),
            mode="overwrite",
            configuration=_CDF_CONFIG,
            partition_by=partition_list,
            writer_properties=WriterProperties(),
        )
