"""Integration tests for @incremental with a Polars Cloud ComputeContext.

These tests require:
  - A valid Polars Cloud workspace (``POLARS_CLOUD_WORKSPACE`` env var)
  - AWS credentials and ``MUNICH_CYCLING_BUCKET`` env var for the Delta target
    and Parquet staging path

Skipped automatically when ``POLARS_CLOUD_WORKSPACE`` is not set so they never block the
local / CI test suite.
"""

from __future__ import annotations

import os
from typing import cast

import polars as pl
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("POLARS_CLOUD_WORKSPACE"),
    reason="POLARS_CLOUD_WORKSPACE not set — skipping Polars Cloud integration tests",
)


@pytest.fixture(scope="module")
def ctx():
    import polars_cloud as pc

    pc.authenticate()
    return pc.ComputeContext(
        workspace=os.environ["POLARS_CLOUD_WORKSPACE"],
        instance_type=os.environ.get("PC_INSTANCE_TYPE", "t4g.micro"),
    )


@pytest.fixture(scope="module")
def bucket():
    return os.environ.get("MUNICH_CYCLING_BUCKET", "data-warehousing-with-polars")


@pytest.fixture(scope="module")
def run_id():
    """Unique prefix so each test-module invocation uses isolated S3 paths."""
    import uuid

    return uuid.uuid4().hex[:8]


@pytest.mark.slow
def test_incremental_remote_append(ctx, bucket, run_id):
    """@incremental with compute_context executes the transform on the cluster
    and appends results to a Delta table via Parquet staging.

    Source data is written directly to S3 so the LazyFrame plan only references
    cloud storage and is eligible for remote execution.
    """
    from data_warehousing_with_polars import incremental

    source = f"s3://{bucket}/tests/{run_id}/cloud_append/source"
    target = f"s3://{bucket}/tests/{run_id}/delta/cloud_append"
    staging = f"s3://{bucket}/tests/{run_id}/staging/cloud_append/"

    pl.DataFrame({"id": [1, 2, 3], "value": [1.0, 2.0, 3.0]}).write_parquet(
        f"{source}/batch.parquet"
    )

    @incremental(source=source, target=target, compute_context=ctx, staging=staging)
    def pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns((pl.col("value") * 10).alias("value"))

    processed = pipeline.run()
    assert len(processed) == 1

    result = pl.scan_delta(target).collect()
    result = cast(pl.DataFrame, result)

    assert set(result["id"].to_list()) == {1, 2, 3}
    assert set(result["value"].to_list()) == {10.0, 20.0, 30.0}

    # Second run: watermark prevents reprocessing.
    assert pipeline.run() == []


@pytest.mark.slow
def test_incremental_remote_merge(ctx, bucket, run_id):
    """@incremental with compute_context and merge_on upserts correctly."""
    from data_warehousing_with_polars import incremental

    source = f"s3://{bucket}/tests/{run_id}/cloud_merge/source"
    target = f"s3://{bucket}/tests/{run_id}/delta/cloud_merge"
    staging = f"s3://{bucket}/tests/{run_id}/staging/cloud_merge/"

    pl.DataFrame({"id": [1, 2], "label": ["a", "b"]}).write_parquet(f"{source}/v1.parquet")

    @incremental(
        source=source,
        target=target,
        merge_on="id",
        compute_context=ctx,
        staging=staging,
    )
    def pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    pipeline.run()

    # Write an update + a new row.
    pl.DataFrame({"id": [2, 3], "label": ["B", "c"]}).write_parquet(f"{source}/v2.parquet")
    pipeline.run()

    result = pl.scan_delta(target).collect()
    result = cast(pl.DataFrame, result)
    assert result.sort("id")["label"].to_list() == ["a", "B", "c"]
