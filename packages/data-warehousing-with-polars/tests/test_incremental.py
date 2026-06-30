"""Tests for incremental module."""

from pathlib import Path
from typing import Literal, cast

import polars as pl
import pytest
from data_warehousing_with_polars.incremental import (
    Batch,
    FrameSource,
    IncrementalPipeline,
    QuerySource,
    Source,
    _DeltaCdfSource,
    _DirSource,
    _list_local_files,
    from_frame,
    from_query,
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
    assert len(test_pipeline._sources) == 2
    assert all(isinstance(s, _DirSource) for s in test_pipeline._sources)


def test_incremental_delta_format():
    """Test that file_format='delta' wraps the source as a Delta CDF source."""

    @incremental(
        source="/tmp/delta_source/",
        target="/tmp/output/",
        merge_on="id",
        file_format="delta",
    )
    def test_pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    assert test_pipeline.file_format == "delta"
    assert len(test_pipeline._sources) == 1
    assert isinstance(test_pipeline._sources[0], _DeltaCdfSource)


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
    assert isinstance(test_pipeline._sources[0], _DirSource)
    assert test_pipeline._sources[0]._suffixes == (".csv",)


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
    assert isinstance(test_pipeline._sources[0], _DirSource)
    assert test_pipeline._sources[0]._suffixes == (".ndjson",)


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


# ── Source protocol tests ─────────────────────────────────────────────────────


def test_from_query_returns_query_source():
    src = from_query(lambda since: None, cursor_on="updated_at")
    assert isinstance(src, QuerySource)
    assert isinstance(src, Source)


def test_from_frame_returns_frame_source():
    src = from_frame(lambda: pl.DataFrame({"id": [1]}).lazy())
    assert isinstance(src, FrameSource)
    assert isinstance(src, Source)


def test_query_source_poll_returns_none_when_fn_returns_none():
    src = from_query(lambda since: None, cursor_on="ts")
    assert src.poll(None) is None
    assert src.poll("2024-01") is None


def test_query_source_poll_returns_batch_with_cursor():
    data = pl.DataFrame({"id": [1, 2], "ts": ["2024-01-01", "2024-01-02"]})
    src = from_query(lambda since: data.lazy(), cursor_on="ts")
    batch = src.poll(None)
    assert batch is not None
    assert isinstance(batch, Batch)
    assert batch.cursor == "2024-01-02"
    assert isinstance(batch.frame, pl.LazyFrame)


def test_query_source_cursor_filters_correctly():
    calls: list[object] = []

    def fetch(since: object) -> pl.LazyFrame | None:
        calls.append(since)
        if since == "2024-01-02":
            return None
        return pl.DataFrame({"id": [1], "ts": ["2024-01-02"]}).lazy()

    src = from_query(fetch, cursor_on="ts")
    batch = src.poll(None)
    assert batch is not None
    assert batch.cursor == "2024-01-02"

    result = src.poll(batch.cursor)
    assert result is None
    assert calls == [None, "2024-01-02"]


def test_frame_source_always_returns_batch():
    df = pl.DataFrame({"id": [1, 2]})
    src = from_frame(lambda: df.lazy())

    b1 = src.poll(None)
    b2 = src.poll("anything")
    assert b1 is not None
    assert b2 is not None
    assert b1.cursor is None
    assert b2.cursor is None


def test_incremental_accepts_custom_source():
    src = from_query(lambda since: None, cursor_on="ts")

    @incremental(source=src, target="/tmp/output/")
    def pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    assert isinstance(pipeline, IncrementalPipeline)
    assert pipeline._sources == [src]


def test_incremental_custom_source_run_returns_empty_when_up_to_date(tmp_path):
    src = from_query(lambda since: None, cursor_on="ts")

    @incremental(source=src, target=str(tmp_path / "target"))
    def pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    result = pipeline.run()
    assert result == []


def test_incremental_custom_source_run_dry_run(tmp_path):
    data = pl.DataFrame({"id": [1], "ts": ["2024-06"]})
    src = from_query(lambda since: data.lazy(), cursor_on="ts")

    @incremental(source=src, target=str(tmp_path / "target"))
    def pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    result = pipeline.run(dry_run=True)
    assert result == ["2024-06"]
    assert not (tmp_path / "target").exists()


def test_incremental_from_query_end_to_end(tmp_path):
    """from_query source ingests data, saves cursor, skips on second run."""
    call_count = [0]

    def fetch(since: object) -> pl.LazyFrame | None:
        call_count[0] += 1
        if since == "2024-01-03":
            return None
        return (
            pl.DataFrame(
                {
                    "id": [1, 2, 3],
                    "ts": ["2024-01-01", "2024-01-02", "2024-01-03"],
                    "value": [10.0, 20.0, 30.0],
                }
            )
            .lazy()
            .filter(pl.col("ts") > (since or "1970-01-01"))
        )

    @incremental(
        source=from_query(fetch, cursor_on="ts"),
        target=str(tmp_path / "target"),
        merge_on="id",
    )
    def pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    result = pipeline.run()
    assert result == ["2024-01-03"]
    assert call_count[0] == 1

    result2 = pipeline.run()
    assert result2 == []
    assert call_count[0] == 2


# ── Fan-in across source types ─────────────────────────────────────────────────


def _write_delta_cdf(path, df: pl.DataFrame, mode: Literal["append", "overwrite"]) -> None:
    """Write *df* to a Delta table at *path* with Change Data Feed enabled."""
    from deltalake import write_deltalake

    write_deltalake(
        str(path),
        df.to_arrow(),
        mode=mode,
        configuration={"delta.enableChangeDataFeed": "true"},
    )


def test_fan_in_dirs_end_to_end(tmp_path):
    """Two directory sources fan in; the transform receives one frame per source."""
    src_a = tmp_path / "a"
    src_b = tmp_path / "b"
    src_a.mkdir()
    src_b.mkdir()
    pl.DataFrame({"id": [1], "v": [10.0]}).write_parquet(src_a / "f.parquet")
    pl.DataFrame({"id": [2], "v": [20.0]}).write_parquet(src_b / "f.parquet")

    @incremental(source=[str(src_a), str(src_b)], target=str(tmp_path / "t"), merge_on="id")
    def pipe(*lfs: pl.LazyFrame) -> pl.LazyFrame:
        assert len(lfs) == 2
        return pl.concat(lfs).select("id", "v")

    result = pipe.run()
    assert len(result) == 2  # one new file per source

    out = cast(pl.DataFrame, pl.scan_delta(str(tmp_path / "t")).collect())
    assert set(out["id"].to_list()) == {1, 2}

    assert pipe.run() == []  # nothing new on rerun


def test_fan_in_delta_sources(tmp_path):
    """Two Delta sources fan in with independent per-source version cursors."""
    src_a = tmp_path / "da"
    src_b = tmp_path / "db"
    _write_delta_cdf(src_a, pl.DataFrame({"id": [1], "v": [10]}), "overwrite")
    _write_delta_cdf(src_b, pl.DataFrame({"id": [2], "v": [20]}), "overwrite")

    @incremental(
        source=[str(src_a), str(src_b)],
        target=str(tmp_path / "t"),
        merge_on="id",
        file_format="delta",
    )
    def pipe(*lfs: pl.LazyFrame) -> pl.LazyFrame:
        return pl.concat([lf.select("id", "v") for lf in lfs])

    result = pipe.run()
    assert len(result) == 2  # both sources had an initial version
    assert set(
        cast(pl.DataFrame, pl.scan_delta(str(tmp_path / "t")).collect())["id"].to_list()
    ) == {1, 2}

    assert pipe.run() == []  # neither source advanced

    # Advance only src_a — only its frame should arrive on the next run.
    _write_delta_cdf(src_a, pl.DataFrame({"id": [3], "v": [30]}), "append")
    result3 = pipe.run()
    assert len(result3) == 1
    assert set(
        cast(pl.DataFrame, pl.scan_delta(str(tmp_path / "t")).collect())["id"].to_list()
    ) == {1, 2, 3}


def test_fan_in_custom_sources(tmp_path):
    """Two custom Sources fan in; an idle source is omitted from the transform args."""
    a_calls: list[object] = []

    def fetch_a(since: object) -> pl.LazyFrame | None:
        a_calls.append(since)
        nxt = "2024-02" if since is None else "2024-03"
        if since == "2024-03":
            return None
        return pl.DataFrame({"id": [len(a_calls)], "ts": [nxt]}).lazy()

    def fetch_b(since: object) -> pl.LazyFrame | None:
        if since is not None:
            return None  # only ever yields on the first run
        return pl.DataFrame({"id": [100], "ts": ["2024-02"]}).lazy()

    @incremental(
        source=[from_query(fetch_a, cursor_on="ts"), from_query(fetch_b, cursor_on="ts")],
        target=str(tmp_path / "t"),
        merge_on="id",
    )
    def pipe(*lfs: pl.LazyFrame) -> pl.LazyFrame:
        return pl.concat([lf.select("id", "ts") for lf in lfs])

    r1 = pipe.run()
    assert len(r1) == 2  # both sources produced data

    r2 = pipe.run()  # b is idle now, a still advances
    assert len(r2) == 1

    out = cast(pl.DataFrame, pl.scan_delta(str(tmp_path / "t")).collect())
    assert 100 in out["id"].to_list()


def test_fan_in_mixed_dir_and_custom(tmp_path):
    """A directory source and a custom Source fan in together."""
    src_dir = tmp_path / "dir"
    src_dir.mkdir()
    pl.DataFrame({"id": [1], "ts": ["2024-01"]}).write_parquet(src_dir / "f.parquet")

    def fetch(since: object) -> pl.LazyFrame | None:
        if since == "2024-05":
            return None
        return pl.DataFrame({"id": [2], "ts": ["2024-05"]}).lazy()

    @incremental(
        source=[str(src_dir), from_query(fetch, cursor_on="ts")],
        target=str(tmp_path / "t"),
        merge_on="id",
    )
    def pipe(*lfs: pl.LazyFrame) -> pl.LazyFrame:
        return pl.concat([lf.select("id", "ts") for lf in lfs])

    result = pipe.run()
    assert len(result) == 2  # one new file + one custom cursor

    out = cast(pl.DataFrame, pl.scan_delta(str(tmp_path / "t")).collect())
    assert set(out["id"].to_list()) == {1, 2}


# ── by_partition tests ────────────────────────────────────────────────────────


def test_by_partition_without_partition_by_raises() -> None:
    with pytest.raises(ValueError, match="partition_by"):

        @incremental(source="/tmp/in/", target="/tmp/out/", merge_on="id", by_partition=True)
        def pipe(lf: pl.LazyFrame) -> pl.LazyFrame:
            return lf


def test_by_partition_stored_as_attribute() -> None:
    @incremental(
        source="/tmp/in/",
        target="/tmp/out/",
        merge_on="id",
        partition_by="region",
        by_partition=True,
    )
    def pipe(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    assert pipe.by_partition is True


def test_by_partition_fn_called_once_per_partition(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    pl.DataFrame(
        {"id": [1, 2, 3, 4], "region": ["EU", "EU", "US", "US"], "v": [10, 20, 30, 40]}
    ).write_parquet(src / "data.parquet")

    call_log: list[str] = []

    @incremental(
        source=str(src),
        target=str(tmp_path / "tgt"),
        merge_on="id",
        partition_by="region",
        by_partition=True,
    )
    def pipe(lf: pl.LazyFrame) -> pl.LazyFrame:
        df = cast(pl.DataFrame, lf.collect())
        call_log.extend(df["region"].unique().to_list())
        return df.lazy()

    pipe.run()

    assert len(call_log) == 2
    assert set(call_log) == {"EU", "US"}


def test_by_partition_result_matches_non_partitioned(tmp_path: Path) -> None:
    data = pl.DataFrame(
        {"id": [1, 2, 3, 4], "region": ["EU", "EU", "US", "US"], "v": [10, 20, 30, 40]}
    )
    for name in ("src_a", "src_b"):
        d = tmp_path / name
        d.mkdir()
        data.write_parquet(d / "data.parquet")

    @incremental(
        source=str(tmp_path / "src_a"),
        target=str(tmp_path / "by_part"),
        merge_on="id",
        partition_by="region",
        by_partition=True,
    )
    def pipe_a(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.select("id", "region", "v")

    @incremental(
        source=str(tmp_path / "src_b"),
        target=str(tmp_path / "normal"),
        merge_on="id",
        partition_by="region",
    )
    def pipe_b(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.select("id", "region", "v")

    pipe_a.run()
    pipe_b.run()

    result_a = cast(pl.DataFrame, pl.scan_delta(str(tmp_path / "by_part")).collect()).sort("id")
    result_b = cast(pl.DataFrame, pl.scan_delta(str(tmp_path / "normal")).collect()).sort("id")
    assert result_a.equals(result_b)


def test_by_partition_scd2(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    pl.DataFrame(
        {"id": [1, 2, 3, 4], "region": ["EU", "EU", "US", "US"], "name": ["A", "B", "C", "D"]}
    ).write_parquet(src / "v1.parquet")

    @incremental(
        source=str(src),
        target=str(tmp_path / "tgt"),
        merge_on="id",
        partition_by="region",
        scd_type=2,
        by_partition=True,
    )
    def pipe(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    pipe.run()
    result = cast(pl.DataFrame, pl.scan_delta(str(tmp_path / "tgt")).collect())
    assert len(result) == 4
    assert result["is_current"].all()

    pl.DataFrame({"id": [1, 2], "region": ["EU", "EU"], "name": ["A2", "B2"]}).write_parquet(
        src / "v2.parquet"
    )

    pipe.run()
    result2 = cast(pl.DataFrame, pl.scan_delta(str(tmp_path / "tgt")).collect())
    # 2 closed EU + 2 unchanged US + 2 new current EU = 6 rows
    assert len(result2) == 6
    eu_current = result2.filter(pl.col("region") == "EU").filter(pl.col("is_current"))
    assert len(eu_current) == 2
    assert set(eu_current["name"].to_list()) == {"A2", "B2"}
    us_current = result2.filter(pl.col("region") == "US").filter(pl.col("is_current"))
    assert len(us_current) == 2


def test_by_partition_scd4(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    pl.DataFrame(
        {"id": [1, 2, 3, 4], "region": ["EU", "EU", "US", "US"], "name": ["A", "B", "C", "D"]}
    ).write_parquet(src / "v1.parquet")

    @incremental(
        source=str(src),
        target=str(tmp_path / "tgt"),
        history_target=str(tmp_path / "hist"),
        merge_on="id",
        partition_by="region",
        scd_type=4,
        by_partition=True,
    )
    def pipe(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    pipe.run()
    assert len(cast(pl.DataFrame, pl.scan_delta(str(tmp_path / "tgt")).collect())) == 4

    pl.DataFrame({"id": [1, 2], "region": ["EU", "EU"], "name": ["A2", "B2"]}).write_parquet(
        src / "v2.parquet"
    )

    pipe.run()
    current = cast(pl.DataFrame, pl.scan_delta(str(tmp_path / "tgt")).collect())
    assert len(current) == 4
    eu = current.filter(pl.col("region") == "EU").sort("id")
    assert eu["name"].to_list() == ["A2", "B2"]
    hist = cast(pl.DataFrame, pl.scan_delta(str(tmp_path / "hist")).collect())
    assert len(hist) == 2
    assert set(hist["name"].to_list()) == {"A", "B"}
