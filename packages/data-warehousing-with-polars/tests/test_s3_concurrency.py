"""Stress tests proving out concurrent-commit behaviour against real S3.

These tests require ``AWS_S3_TEST_BUCKET`` (and AWS credentials) to be set, and
are skipped otherwise so they never block the local test suite or CI. They get
their own ``poe test_concurrency`` task — never part of ``poe qa``/``poe test``,
and never wired into any CI workflow.

Every target used lives under ``s3://{bucket}/tests/{run_id}/...`` and is
deleted at teardown regardless of outcome.

Every existing concurrency test for ``by_partition`` (see
``test_incremental.py``'s ``test_by_partition_workers_*``) writes to a local
filesystem target, whose safety mechanism (atomic exclusive file creation) is
different from S3's ETag-based conditional PUT. The tests here exercise the
same code paths against a real bucket instead, asserting on write correctness
(no lost or duplicated commits) rather than timing overlap.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Generator
from pathlib import Path
from typing import cast

import polars as pl
import pytest
from data_warehousing_with_polars import incremental
from data_warehousing_with_polars.incremental import _DeltaCdfSource

pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("AWS_S3_TEST_BUCKET"),
        reason="AWS_S3_TEST_BUCKET not set — skipping S3 concurrency stress tests",
    ),
    pytest.mark.slow,
]


def _write_partitioned_measurements(
    src: Path,
    n_partitions: int,
    rows_per_partition: int,
    suffix: str = "00001",
    value_offset: float = 0.0,
) -> None:
    """Write partitioned parquet with (measurement, channel, value) columns.

    Adapted from ``utils/sample_data.write_partitioned_measurements``: ``channel``
    embeds the partition index, so it's unique *across the whole table*, not just
    within one partition — required because ``merge_on`` alone (without
    ``partition_by`` folded in) is the row identity for the SCD2/SCD4 sinks
    (``_sink_scd2``/``_sink_scd4`` in ``scd.py``), unlike the plain upsert path
    (``_sink_target``/``_upsert_overwrite``), which folds ``partition_by`` into the
    join columns itself. A channel value that recurred across partitions would
    make ``_sink_scd4``'s unfiltered archive-table scan match every partition
    sharing that value, not just the current one — a test-data bug, not a
    production one.

    ``suffix`` names the file within each partition dir, so calling this again
    with a different suffix adds a genuinely new file for a later run to pick up.
    """
    for p in range(n_partitions):
        measurement = f"measurement_{p}"
        partition_dir = src / f"measurement={measurement}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        df = pl.DataFrame(
            {
                "measurement": [measurement] * rows_per_partition,
                "channel": [f"{p:04d}_{i:06d}" for i in range(rows_per_partition)],
                "value": [value_offset + float(i) for i in range(rows_per_partition)],
            }
        )
        df.write_parquet(partition_dir / f"{suffix}.parquet")


@pytest.fixture
def bucket() -> str:
    return os.environ["AWS_S3_TEST_BUCKET"]


@pytest.fixture
def s3_root(bucket: str) -> Generator[str, None, None]:
    """Unique ``s3://{bucket}/tests/{run_id}`` prefix, recursively deleted at teardown."""
    from pyarrow import fs as pa_fs

    root = f"s3://{bucket}/tests/{uuid.uuid4().hex[:8]}"
    try:
        yield root
    finally:
        filesystem, base_path = pa_fs.FileSystem.from_uri(root)
        try:
            filesystem.delete_dir(base_path)
        except FileNotFoundError:
            pass


def test_by_partition_append_concurrent_commits(tmp_path: Path, s3_root: str) -> None:
    """Plain (append-only) ``by_partition`` run against a real S3 target: every
    partition's commit lands — none lost to conditional-PUT conflicts."""
    n_partitions = 16
    rows_per_partition = 200
    target = f"{s3_root}/target"
    src = tmp_path / "src"
    src.mkdir()
    _write_partitioned_measurements(src, n_partitions, rows_per_partition)

    @incremental(
        source=str(src),
        target=target,
        partition_by="measurement",
        by_partition=True,
        by_partition_workers=8,
    )
    def pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    pipeline.run()

    result = cast(pl.DataFrame, pl.scan_delta(target).collect())
    assert len(result) == n_partitions * rows_per_partition
    assert set(result["measurement"].unique().to_list()) == {
        f"measurement_{p}" for p in range(n_partitions)
    }


def test_by_partition_upsert_concurrent_commits(tmp_path: Path, s3_root: str) -> None:
    """Concurrent per-partition anti-join + ``replaceWhere`` upserts against S3
    update every row exactly once — no lost updates, no duplicates."""
    n_partitions = 16
    rows_per_partition = 200
    target = f"{s3_root}/target"
    src = tmp_path / "src"
    src.mkdir()
    _write_partitioned_measurements(src, n_partitions, rows_per_partition)

    @incremental(
        source=str(src),
        target=target,
        merge_on="channel",
        partition_by="measurement",
        by_partition=True,
        by_partition_workers=8,
    )
    def pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    pipeline.run()  # creates the table

    # Same (measurement, channel) keys, new values — every partition's commit is
    # now an ordinary (non-creating) concurrent upsert against the shared log.
    _write_partitioned_measurements(
        src, n_partitions, rows_per_partition, suffix="00002", value_offset=1_000_000.0
    )
    pipeline.run()

    result = cast(pl.DataFrame, pl.scan_delta(target).collect())
    assert len(result) == n_partitions * rows_per_partition
    assert cast(float, result["value"].min()) >= 1_000_000.0
    assert result.select(["measurement", "channel"]).is_duplicated().sum() == 0


def test_by_partition_scd2_concurrent_commits(tmp_path: Path, s3_root: str) -> None:
    """Concurrent per-partition SCD2 rewrites (close+append in one commit) against
    S3 preserve the valid_from/valid_to/is_current invariants under real contention."""
    n_partitions = 16
    rows_per_partition = 50
    target = f"{s3_root}/target"
    src = tmp_path / "src"
    src.mkdir()
    _write_partitioned_measurements(src, n_partitions, rows_per_partition)

    @incremental(
        source=str(src),
        target=target,
        merge_on="channel",
        partition_by="measurement",
        scd_type=2,
        by_partition=True,
        by_partition_workers=8,
    )
    def pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    pipeline.run()

    _write_partitioned_measurements(
        src, n_partitions, rows_per_partition, suffix="00002", value_offset=1_000_000.0
    )
    pipeline.run()

    result = cast(pl.DataFrame, pl.scan_delta(target).collect())
    assert len(result) == 2 * n_partitions * rows_per_partition

    current = result.filter(pl.col("is_current"))
    assert len(current) == n_partitions * rows_per_partition
    assert cast(float, current["value"].min()) >= 1_000_000.0
    assert current.select(["measurement", "channel"]).is_duplicated().sum() == 0

    closed = result.filter(~pl.col("is_current"))
    assert len(closed) == n_partitions * rows_per_partition
    assert cast(int, closed["valid_to"].null_count()) == 0


def test_by_partition_scd4_concurrent_commits(tmp_path: Path, s3_root: str) -> None:
    """Concurrent per-partition SCD4 archive+upsert commits against two S3 tables
    (current + history, both guarded by the same ``creation_lock``) stay correct."""
    n_partitions = 16
    rows_per_partition = 50
    target = f"{s3_root}/target"
    history_target = f"{s3_root}/history"
    src = tmp_path / "src"
    src.mkdir()
    _write_partitioned_measurements(src, n_partitions, rows_per_partition)

    @incremental(
        source=str(src),
        target=target,
        history_target=history_target,
        merge_on="channel",
        partition_by="measurement",
        scd_type=4,
        by_partition=True,
        by_partition_workers=8,
    )
    def pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    pipeline.run()

    _write_partitioned_measurements(
        src, n_partitions, rows_per_partition, suffix="00002", value_offset=1_000_000.0
    )
    pipeline.run()

    current = cast(pl.DataFrame, pl.scan_delta(target).collect())
    assert len(current) == n_partitions * rows_per_partition
    assert cast(float, current["value"].min()) >= 1_000_000.0
    assert current.select(["measurement", "channel"]).is_duplicated().sum() == 0

    history = cast(pl.DataFrame, pl.scan_delta(history_target).collect())
    assert len(history) == n_partitions * rows_per_partition
    assert cast(int, history["superseded_at"].null_count()) == 0


def test_by_partition_cdf_source_concurrent_commits(tmp_path: Path, s3_root: str) -> None:
    """Reading a Change Data Feed from an S3 Delta source, one predicate-scoped
    ``load_cdf`` call per partition, and concurrently committing the sink to
    another S3 target both survive real conditional-PUT contention."""
    from deltalake import write_deltalake

    n_partitions = 16
    rows_per_partition = 200
    source = f"{s3_root}/cdf_source"
    target = f"{s3_root}/target"

    def _seed(offset: float) -> pl.DataFrame:
        return pl.concat(
            [
                pl.DataFrame(
                    {
                        "measurement": [f"measurement_{p}"] * rows_per_partition,
                        "value": [offset + float(i) for i in range(rows_per_partition)],
                    }
                )
                for p in range(n_partitions)
            ]
        )

    write_deltalake(
        source,
        _seed(0.0).to_arrow(),
        mode="overwrite",
        partition_by=["measurement"],
        configuration={"delta.enableChangeDataFeed": "true"},
    )

    @incremental(
        source=source,
        target=target,
        file_format="delta",
        partition_by="measurement",
        by_partition=True,
        by_partition_workers=8,
    )
    def pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    pipeline.run()  # first run: full scan, establishes the CDF cursor

    write_deltalake(
        source, _seed(1_000_000.0).to_arrow(), mode="append", partition_by=["measurement"]
    )
    pipeline.run()  # second run: per-partition load_cdf + concurrent commits to target

    result = cast(pl.DataFrame, pl.scan_delta(target).collect())
    assert len(result) == 2 * n_partitions * rows_per_partition
    assert result.filter(pl.col("value") >= 1_000_000.0).height == n_partitions * rows_per_partition


def test_by_partition_fan_in_concurrent_commits(tmp_path: Path, s3_root: str) -> None:
    """Fanning in a local file source and an S3 CDF source, then concurrently
    committing the combined per-partition result to an S3 target, loses nothing."""
    from deltalake import write_deltalake

    n_partitions = 16
    rows_per_partition = 100
    target = f"{s3_root}/target"
    cdf_source = f"{s3_root}/cdf_source"

    file_src = tmp_path / "src"
    file_src.mkdir()
    _write_partitioned_measurements(file_src, n_partitions, rows_per_partition)

    def _cdf_seed(offset: float) -> pl.DataFrame:
        return pl.concat(
            [
                pl.DataFrame(
                    {
                        "measurement": [f"measurement_{p}"] * rows_per_partition,
                        "value": [offset + float(i) for i in range(rows_per_partition)],
                    }
                )
                for p in range(n_partitions)
            ]
        )

    write_deltalake(
        cdf_source,
        _cdf_seed(0.0).to_arrow(),
        mode="overwrite",
        partition_by=["measurement"],
        configuration={"delta.enableChangeDataFeed": "true"},
    )

    @incremental(
        source=[str(file_src), _DeltaCdfSource(cdf_source)],
        target=target,
        partition_by="measurement",
        by_partition=True,
        by_partition_workers=8,
    )
    def pipeline(lf_file: pl.LazyFrame, lf_cdf: pl.LazyFrame) -> pl.LazyFrame:
        return pl.concat(
            [lf_file.select("measurement", "value"), lf_cdf.select("measurement", "value")],
            how="diagonal_relaxed",
        )

    pipeline.run()  # both sources active: full scan + full CDF read

    write_deltalake(
        cdf_source, _cdf_seed(1_000_000.0).to_arrow(), mode="append", partition_by=["measurement"]
    )
    _write_partitioned_measurements(
        file_src, n_partitions, rows_per_partition, suffix="00002", value_offset=2_000_000.0
    )
    pipeline.run()  # both sources active again, concurrent commits per partition

    result = cast(pl.DataFrame, pl.scan_delta(target).collect())
    assert len(result) == 4 * n_partitions * rows_per_partition


def test_by_partition_high_concurrency_on_existing_table(tmp_path: Path, s3_root: str) -> None:
    """A much higher ``by_partition_workers``/partition count than the tests above,
    run against an already-existing S3 table, proves delta-rs's own commit retry —
    not ``creation_lock``, which only ever guards the very first write — holds up
    under heavier real concurrency."""
    target = f"{s3_root}/target"
    src = tmp_path / "src"
    src.mkdir()
    rows_per_partition = 50

    @incremental(
        source=str(src),
        target=target,
        partition_by="measurement",
        by_partition=True,
        by_partition_workers=32,
    )
    def pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    # First batch: small, establishes the table.
    _write_partitioned_measurements(src, 4, rows_per_partition, suffix="00001")
    pipeline.run()

    # Second batch: far more partitions than tests 1-6 use, all new files — a
    # genuinely high fan-out of concurrent commits against the existing table.
    n_partitions = 48
    _write_partitioned_measurements(src, n_partitions, rows_per_partition, suffix="00002")
    pipeline.run()

    result = cast(pl.DataFrame, pl.scan_delta(target).collect())
    assert len(result) == (4 + n_partitions) * rows_per_partition
    assert len(result["measurement"].unique()) == n_partitions
