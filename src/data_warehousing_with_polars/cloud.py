"""
cloud.py

Remote execution path for Polars Cloud: runs the transform on a cluster,
reads staged Parquet output, and drives the Delta merge locally.
"""

from __future__ import annotations

import logging

import polars as pl
from deltalake import DeltaTable, WriterProperties, write_deltalake
from deltalake.exceptions import TableNotFoundError

logger = logging.getLogger(__name__)

_CDF_CONFIG = {"delta.enableChangeDataFeed": "true"}


def _sink_target_remote(
    target: str,
    lf: pl.LazyFrame,
    merge_on: str | list[str] | None,
    context: object,
    staging: str,
    partition_by: str | list[str] | None = None,
) -> None:
    """Execute *lf* on a Polars Cloud cluster, then merge the staged output into *target* locally.

    Args:
        target:       Output Delta table path.
        lf:           LazyFrame to execute remotely.
        merge_on:     Upsert key column(s), or ``None`` for append-only.
        context:      ``polars_cloud.ComputeContext`` specifying cluster hardware.
        staging:      S3 path for temporary Parquet output. Not cleaned up by this function.
        partition_by: Partition column(s).
    """
    # Phase 1: Remote execution.
    lf.remote(context).sink_parquet(staging)  # type: ignore[attr-defined]
    logger.info("Remote execution complete; reading staged results from %s", staging)

    # Phase 2: Local Delta merge.
    df = pl.read_parquet(staging)
    keys = [merge_on] if isinstance(merge_on, str) else (list(merge_on) if merge_on else [])
    partition_list = (
        [partition_by]
        if isinstance(partition_by, str)
        else (list(partition_by) if partition_by else None)
    )

    first_write = False
    try:
        DeltaTable(target)
    except TableNotFoundError:
        first_write = True

    if not keys:
        if first_write:
            write_deltalake(
                target,
                df.to_arrow(),
                mode="overwrite",
                configuration=_CDF_CONFIG,
                partition_by=partition_list,
                writer_properties=WriterProperties(),
            )
        else:
            write_deltalake(
                target, df.to_arrow(), mode="append", writer_properties=WriterProperties()
            )
        return

    if first_write:
        write_deltalake(
            target,
            df.to_arrow(),
            mode="overwrite",
            configuration=_CDF_CONFIG,
            partition_by=partition_list,
            writer_properties=WriterProperties(),
        )
        return

    predicate = " AND ".join(f"target.{k} = source.{k}" for k in keys)
    dt = DeltaTable(target)
    (
        dt.merge(df.to_arrow(), predicate=predicate, source_alias="source", target_alias="target")
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute()
    )
