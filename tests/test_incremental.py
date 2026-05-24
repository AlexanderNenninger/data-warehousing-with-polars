"""Tests for incremental module."""

import polars as pl
import pytest

from data_warehousing_with_polars.incremental import (
    IncrementalPipeline,
    _list_local_files,
    incremental,
)


def test_list_paths_local(tmp_path):
    """Test that _list_paths finds files in a local directory."""
    import polars as pl

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    pl.DataFrame({"id": [1, 2], "value": [10.0, 20.0]}).write_parquet(input_dir / "file1.parquet")
    pl.DataFrame({"id": [3, 4], "value": [30.0, 40.0]}).write_parquet(input_dir / "file2.parquet")
    (input_dir / "ignore.csv").write_text("id,value\n5,50.0")

    paths = _list_local_files(str(input_dir), (".parquet",))

    assert len(paths) == 2
    assert all(p.endswith(".parquet") for p in paths)


def test_list_paths_empty_dir(tmp_path):
    """Test that _list_local_files returns [] for an empty directory."""
    (tmp_path / "empty").mkdir()
    paths = _list_local_files(str(tmp_path / "empty"), (".parquet",))
    assert paths == []


def test_list_paths_nonexistent(tmp_path):
    """Test that _list_local_files returns [] for a non-existent directory."""
    paths = _list_local_files(str(tmp_path / "does_not_exist"), (".parquet",))
    assert paths == []


def test_incremental_local_path(tmp_path):
    """Test that the incremental decorator accepts local paths."""

    @incremental(
        source=str(tmp_path / "input"),
        target=str(tmp_path / "output"),
    )
    def test_pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    assert isinstance(test_pipeline, IncrementalPipeline)
    assert test_pipeline.source == str(tmp_path / "input")


def test_incremental_fan_in_source():
    """Test that a list of source paths is stored correctly."""

    @incremental(
        source=["/tmp/input1/", "/tmp/input2/"],
        target="/tmp/output/",
    )
    def test_pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    assert test_pipeline.source == ["/tmp/input1/", "/tmp/input2/"]
    assert test_pipeline._sources == ["/tmp/input1/", "/tmp/input2/"]


def test_incremental_delta_format():
    """Test that file_format='delta' is accepted and has no suffix."""

    @incremental(
        source="/tmp/delta_source/",
        target="/tmp/output/",
        merge_on="id",
        file_format="delta",
    )
    def test_pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    assert test_pipeline.file_format == "delta"
    assert test_pipeline._suffixes == ()


def test_incremental_with_partition_by():
    """Test that partition_by is stored and accessible."""

    @incremental(
        source="/tmp/input/",
        target="/tmp/output/",
        partition_by="date",
    )
    def test_pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    assert test_pipeline.partition_by == "date"


def test_incremental_scd_type_2():
    """Test that scd_type=2 is stored correctly."""

    @incremental(
        source="/tmp/input/",
        target="/tmp/output/",
        merge_on="id",
        scd_type=2,
    )
    def test_pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    assert test_pipeline.scd_type == 2


def test_incremental_scd_type_4_requires_history_target():
    """Test that scd_type=4 requires history_target."""
    with pytest.raises(ValueError, match="history_target"):

        @incremental(
            source="/tmp/input/",
            target="/tmp/output/",
            merge_on="id",
            scd_type=4,
        )
        def test_pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
            return lf


def test_incremental_compute_context_requires_staging():
    """Test that compute_context requires staging."""
    with pytest.raises(ValueError, match="staging"):

        @incremental(
            source="/tmp/input/",
            target="/tmp/output/",
            compute_context=object(),
        )
        def test_pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
            return lf


def test_incremental_decorator():
    """Test that the incremental decorator creates an IncrementalPipeline."""

    @incremental(
        source="s3://test-bucket/input/",
        target="s3://test-bucket/output/",
        merge_on="id",
    )
    def test_pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.filter(pl.col("value") > 0)

    assert isinstance(test_pipeline, IncrementalPipeline)
    assert test_pipeline.source == "s3://test-bucket/input/"
    assert test_pipeline.target == "s3://test-bucket/output/"
    assert test_pipeline.merge_on == "id"


def test_incremental_decorator_callable():
    """Test that the decorated function is still callable."""

    @incremental(
        source="s3://test-bucket/input/",
        target="s3://test-bucket/output/",
    )
    def test_pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.filter(pl.col("value") > 0)

    # Create a test LazyFrame
    lf = pl.DataFrame({"value": [1, 2, -1, 3]}).lazy()
    result = test_pipeline(lf)

    assert isinstance(result, pl.LazyFrame)
    df = result.collect()
    assert len(df) == 3  # Only positive values


def test_incremental_default_watermark():
    """Test that watermark_store defaults to target + /.watermark."""

    @incremental(
        source="s3://test-bucket/input/",
        target="s3://test-bucket/output",
    )
    def test_pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    assert test_pipeline.watermark_store == "s3://test-bucket/output/.watermark"


def test_incremental_custom_watermark():
    """Test that custom watermark_store is respected."""

    @incremental(
        source="s3://test-bucket/input/",
        target="s3://test-bucket/output/",
        watermark_store="s3://test-bucket/custom-watermark/",
    )
    def test_pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    assert test_pipeline.watermark_store == "s3://test-bucket/custom-watermark/"


def test_pipeline_validation_empty_merge_on():
    """Test that empty merge_on list is rejected."""
    with pytest.raises(ValueError, match="must not be an empty list"):

        @incremental(
            source="s3://test-bucket/input/",
            target="s3://test-bucket/output/",
            merge_on=[],
        )
        def test_pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
            return lf


def test_file_format_csv():
    """Test that CSV file format is supported."""

    @incremental(
        source="s3://test-bucket/input/",
        target="s3://test-bucket/output/",
        file_format="csv",
    )
    def test_pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    assert test_pipeline.file_format == "csv"
    assert test_pipeline._suffix == ".csv"


def test_file_format_ndjson():
    """Test that NDJSON file format is supported."""

    @incremental(
        source="s3://test-bucket/input/",
        target="s3://test-bucket/output/",
        file_format="ndjson",
    )
    def test_pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    assert test_pipeline.file_format == "ndjson"
    assert test_pipeline._suffix == ".ndjson"


def test_merge_on_string():
    """Test that merge_on accepts a string."""

    @incremental(
        source="s3://test-bucket/input/",
        target="s3://test-bucket/output/",
        merge_on="id",
    )
    def test_pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    assert test_pipeline.merge_on == "id"


def test_merge_on_list():
    """Test that merge_on accepts a list of strings."""

    @incremental(
        source="s3://test-bucket/input/",
        target="s3://test-bucket/output/",
        merge_on=["id", "timestamp"],
    )
    def test_pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    assert test_pipeline.merge_on == ["id", "timestamp"]


def test_merge_on_none():
    """Test that merge_on accepts None for append-only mode."""

    @incremental(
        source="s3://test-bucket/input/",
        target="s3://test-bucket/output/",
        merge_on=None,
    )
    def test_pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    assert test_pipeline.merge_on is None
