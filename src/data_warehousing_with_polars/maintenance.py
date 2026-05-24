"""
maintenance.py

Table health operations: compaction, Z-ordering, vacuuming, and the run
counter used by compact_every. These have no relationship to data ingestion
— they are concerned with the physical layout of Delta tables, not their
contents — and are separated here so that:

1. They can be scheduled independently of run(). A production setup might
   run pipelines hourly but compact weekly and vacuum daily. Keeping
   maintenance here makes that separation natural rather than forced.

2. The compaction trigger logic (compact_every) needs to persist a run
   counter between process restarts. That state lives alongside the watermark
   store (at <watermark_store>/_runs), but the logic for reading and writing
   it belongs here alongside the compaction it controls, rather than in
   incremental.py where it would clutter the ingestion flow.

Note that vacuum should always be called after compaction, not before. This
module enforces that ordering in maintain() when both are requested.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import polars as pl
from deltalake import DeltaTable, write_deltalake

logger = logging.getLogger(__name__)

_RUNS_SUFFIX = "/_runs"


def _load_run_count(store: str) -> int:
    """Return the current run count from the watermark table.

    Used by the ``compact_every`` trigger to determine whether compaction
    is due on this run. Reads the maximum ``run_count`` value from the
    run-count sub-table at ``<store>/_runs``.

    Args:
        store: Local path to the watermark Delta table.

    Returns:
        The current run count, or ``0`` if no count has been recorded yet.
    """
    runs_store = store.rstrip("/") + _RUNS_SUFFIX
    try:
        result = pl.scan_delta(runs_store).select(pl.col("run_count").max()).collect()
        assert isinstance(result, pl.DataFrame)
        val = result["run_count"][0]
        return int(val) if val is not None else 0
    except Exception:
        return 0


def _save_run_count(store: str, count: int) -> None:
    """Append the updated run count to the watermark table.

    Args:
        store:  Local path to the watermark Delta table.
        count:  The new run count after a successful ``run()`` call.
    """
    runs_store = store.rstrip("/") + _RUNS_SUFFIX
    rows = pl.DataFrame(
        {
            "run_count": [count],
            "updated_at": [datetime.now(timezone.utc)],
        }
    )
    write_deltalake(runs_store, rows, mode="append")


def maintain(
    target: str,
    compact: bool = True,
    z_order_by: str | list[str] | None = None,
    vacuum: bool = True,
    retention_hours: int = 168,
) -> None:
    """Compact and/or vacuum a Delta table.

    When both ``compact`` and ``vacuum`` are requested, compaction runs
    first so that vacuum can remove the pre-compaction files in the same
    pass. Running vacuum before compaction would delete files that
    compaction would have merged, and could destroy time-travel versions
    still within the retention window.

    Z-ordering runs instead of plain compaction when ``z_order_by`` is set.
    It is significantly slower than plain compaction because it reads and
    rewrites all active files, not just the small ones, and should be
    scheduled less frequently (weekly rather than after every batch of runs).

    Args:
        target:          Local path to the Delta table to maintain.
        compact:         Coalesce small files. Ignored when ``z_order_by``
                         is set.
        z_order_by:      Column(s) to Z-order by after compaction. Runs
                         instead of plain compaction.
        vacuum:          Delete files removed more than ``retention_hours``
                         ago.
        retention_hours: Minimum age of files eligible for deletion.
                         Default is 7 days (168 hours). Setting below 168
                         requires ``enforce_retention_duration=False`` on
                         the underlying Delta call, which should only be done
                         in development environments.
    """
    dt = DeltaTable(target)

    if z_order_by is not None:
        cols = [z_order_by] if isinstance(z_order_by, str) else list(z_order_by)
        logger.info("Z-ordering %s by %s", target, cols)
        dt.optimize.z_order(cols)
    elif compact:
        logger.info("Compacting %s", target)
        dt.optimize.compact()

    if vacuum:
        enforce = retention_hours >= 168
        logger.info("Vacuuming %s (retention=%dh, enforce=%s)", target, retention_hours, enforce)
        dt.vacuum(
            retention_hours=retention_hours,
            enforce_retention_duration=enforce,
            dry_run=False,
        )
