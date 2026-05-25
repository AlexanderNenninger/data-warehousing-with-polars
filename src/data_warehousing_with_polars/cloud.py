"""
cloud.py

Remote execution path for Polars Cloud: runs the transform on a cluster,
reads staged Parquet output, and drives the Delta merge locally.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import polars as pl
from deltalake import DeltaTable
from deltalake.exceptions import TableNotFoundError
from deltalake.table import TableMerger

logger = logging.getLogger(__name__)

_CDF_CONFIG = {"delta.enableChangeDataFeed": "true"}


def _sink_target_remote(
    target: str,
    lf: pl.LazyFrame,
    merge_on: str | list[str] | None,
    context: object,
    partition_by: str | list[str] | None = None,
) -> None:
    """Execute *lf* on a Polars Cloud cluster and merge the result into the Delta *target*.

    Uses ``execute()`` to run the transform on the cluster, then ``sink_delta`` to
    stream the result directly into the Delta table — no local collection at any point.

    Args:
        target:       Output Delta table path (S3 URI).
        lf:           LazyFrame to execute remotely.
        merge_on:     Upsert key column(s), or ``None`` for append-only.
        context:      ``polars_cloud.ComputeContext`` specifying cluster hardware.
        partition_by: Partition column(s).
    """
    import polars_cloud as pc

    assert isinstance(context, pc.ClientContext), (
        "Expected a polars_cloud.ComputeContext for remote execution"
    )

    # Execute on the cluster. polars-cloud writes the result to its own temporary
    # storage and returns a QueryResult whose .lazy() is a LazyFrame pointing there.
    query_result = lf.remote(context).execute()  # type: ignore[attr-defined]
    result_lf = query_result.lazy()

    # Phase 2: Delta merge.
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
        # Append: stream directly from cluster temp storage into Delta.
        if first_write:
            result_lf.sink_delta(
                target,
                mode="overwrite",
                delta_write_options={"configuration": _CDF_CONFIG, "partition_by": partition_list},
            )
        else:
            result_lf.sink_delta(target, mode="append")
        return

    # Merge (upsert): stream from cluster temp storage using sink_delta(mode="merge").
    predicate = " AND ".join(f"target.{k} = source.{k}" for k in keys)

    if first_write:
        result_lf.sink_delta(
            target,
            mode="overwrite",
            delta_write_options={"configuration": _CDF_CONFIG, "partition_by": partition_list},
        )
        return

    merger = cast(
        TableMerger,
        result_lf.sink_delta(
            target,
            mode="merge",
            delta_merge_options={
                "predicate": predicate,
                "source_alias": "source",
                "target_alias": "target",
            },
        ),
    )
    merger.when_matched_update_all().when_not_matched_insert_all().execute()
