"""Example usage of the incremental pipeline decorator."""

import logging

import polars as pl
from data_warehousing_with_polars import incremental

# Configure logging to see pipeline progress
logging.basicConfig(level=logging.INFO)


@incremental(
    source="s3://my-bucket/uploads/sensors/",
    target="s3://my-bucket/delta/sensors_clean",
    merge_on="id",
    file_format="parquet",
)
def sensor_pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Process sensor data: filter invalid readings and calculate metrics.

    The pipeline:
    1. Filters out readings with value <= 0
    2. Doubles the value
    3. Flags outliers (value > 50)
    """
    return lf.filter(pl.col("value") > 0).with_columns(
        (pl.col("value") * 2).alias("value_doubled"),
        (pl.col("value") > 50).alias("outlier"),
    )


@incremental(
    source="s3://my-bucket/uploads/logs/",
    target="s3://my-bucket/delta/logs_parsed",
    merge_on=None,  # Append-only mode
    file_format="ndjson",
)
def log_pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Parse and clean log files.

    Append-only mode (merge_on=None) - no deduplication.
    """
    return lf.filter(pl.col("level").is_in(["ERROR", "WARN"])).with_columns(
        pl.col("timestamp").str.to_datetime(),
        pl.col("message").str.to_lowercase().alias("message_lower"),
    )


def main():
    """Run incremental pipelines."""
    print("=== Sensor Pipeline ===")

    # Dry run to preview what would be processed
    new_files = sensor_pipeline.run(dry_run=True)
    print(f"Would process {len(new_files)} new files")

    # Actually process new files
    # sensor_pipeline.run()

    # View processing history
    # print(sensor_pipeline.status())

    # Reset watermark to reprocess everything
    # sensor_pipeline.reset()

    print("\n=== Log Pipeline ===")
    log_pipeline.run(dry_run=True)


if __name__ == "__main__":
    main()
