"""Tests for the FactoryNet streaming pipeline (no network access — every test
that would otherwise hit Hugging Face injects `source_url_fn` to point at local
tmp_path files instead)."""

import polars as pl
import self_supervised_physical_ai.factorynet as factorynet_module
from self_supervised_physical_ai import (
    FACTORYNET_FILES,
    KnownFilesSource,
    SchemaGroup,
    available_metadata_columns,
    build_incremental_pipeline,
    factorynet_source_url,
    group_files_by_schema,
    namespaced_episode_id_expr,
    select_channel_columns,
    source_file_basename,
    split_column_expr,
    split_files_train_test,
    stream_factorynet_to_bucket,
)


def test_factorynet_files_are_unique_parquet_names():
    assert len(FACTORYNET_FILES) == len(set(FACTORYNET_FILES))
    assert all(name.endswith(".parquet") for name in FACTORYNET_FILES)


def test_factorynet_source_url():
    url = factorynet_source_url("voraus_001.parquet")
    assert (
        url
        == "https://huggingface.co/datasets/factorynet/factorynet/resolve/main/data/voraus_001.parquet"
    )


def test_select_channel_columns_picks_populated_numeric_prefixed_columns():
    # Mirrors a real FactoryNet shard: some prefixed columns are all-Null in this
    # source (schema parity with other shards), others are populated; non-prefixed
    # and string columns should never be picked up as channels.
    df = pl.DataFrame(
        {
            "setpoint_pos_0": [1.0, 2.0],
            "feedback_pos_0": [1.0, 2.0],
            "effort_current_0": [None, None],  # all-Null in this source -> excluded
            "ctx_material": ["steel", "steel"],
            "episode_id": ["e1", "e1"],
        }
    )
    channels = select_channel_columns(df.schema)
    assert channels == ["setpoint_pos_0", "feedback_pos_0"]


def test_available_metadata_columns_filters_to_present_columns():
    schema = pl.DataFrame({"episode_id": ["e1"], "time_s": [0.0], "machine_type": ["UR3"]}).schema
    assert available_metadata_columns(schema) == ["episode_id", "time_s", "machine_type"]


def test_split_files_train_test_is_a_partition():
    files = [f"f{i}.parquet" for i in range(10)]
    train, test = split_files_train_test(files, test_size=0.3, seed=0)

    assert set(train) | set(test) == set(files)
    assert set(train) & set(test) == set()
    assert len(test) == 3


def test_split_files_train_test_is_deterministic():
    files = [f"f{i}.parquet" for i in range(10)]
    a = split_files_train_test(files, test_size=0.3, seed=42)
    b = split_files_train_test(files, test_size=0.3, seed=42)
    assert a == b


def _with_source_file(df: pl.DataFrame, path: str) -> pl.DataFrame:
    return df.with_columns(pl.lit(path).alias("_source_file"))


def test_source_file_basename_strips_directory():
    df = _with_source_file(pl.DataFrame({"x": [1]}), "https://example.com/data/voraus_001.parquet")
    out = df.select(source_file_basename().alias("basename"))
    assert out["basename"].to_list() == ["voraus_001.parquet"]


def test_namespaced_episode_id_expr_disambiguates_reused_ids():
    # Same episode_id string from two different source files must not collide.
    df = pl.concat(
        [
            _with_source_file(
                pl.DataFrame({"episode_id": ["VORAUS_0"]}), "https://example.com/voraus_001.parquet"
            ),
            _with_source_file(
                pl.DataFrame({"episode_id": ["VORAUS_0"]}), "https://example.com/voraus_002.parquet"
            ),
        ]
    )
    out = df.select(namespaced_episode_id_expr().alias("episode_id"))
    assert out["episode_id"].to_list() == [
        "voraus_001.parquet/VORAUS_0",
        "voraus_002.parquet/VORAUS_0",
    ]
    assert out["episode_id"].n_unique() == 2


def test_split_column_expr_assigns_by_source_file():
    df = pl.concat(
        [
            _with_source_file(pl.DataFrame({"x": [1]}), "https://example.com/voraus_001.parquet"),
            _with_source_file(pl.DataFrame({"x": [2]}), "https://example.com/voraus_002.parquet"),
            _with_source_file(pl.DataFrame({"x": [3]}), "https://example.com/voraus_003.parquet"),
        ]
    )
    expr = split_column_expr(
        train_files=["voraus_001.parquet", "voraus_002.parquet"],
        test_files=["voraus_003.parquet"],
    )
    out = df.select(expr.alias("split"))
    assert out["split"].to_list() == ["train", "train", "test"]


def test_known_files_source_polls_all_paths_on_first_call(tmp_path):
    file_a = tmp_path / "a.parquet"
    file_b = tmp_path / "b.parquet"
    pl.DataFrame({"x": [1, 2]}).write_parquet(file_a)
    pl.DataFrame({"x": [3]}).write_parquet(file_b)

    source = KnownFilesSource([str(file_a), str(file_b)])
    batch = source.poll(since=None)

    assert batch is not None
    assert sorted(batch.frame.collect()["x"].to_list()) == [1, 2, 3]
    assert batch.cursor == sorted([str(file_a), str(file_b)])

    # Nothing new since batch's cursor -> no batch.
    assert source.poll(since=batch.cursor) is None


def test_known_files_source_returns_only_unprocessed_paths(tmp_path):
    file_a = tmp_path / "a.parquet"
    file_b = tmp_path / "b.parquet"
    pl.DataFrame({"x": [1]}).write_parquet(file_a)
    pl.DataFrame({"x": [2]}).write_parquet(file_b)

    source = KnownFilesSource([str(file_a), str(file_b)])
    batch = source.poll(since=[str(file_a)])

    assert batch is not None
    assert batch.frame.collect()["x"].to_list() == [2]
    assert batch.cursor == sorted([str(file_a), str(file_b)])


def test_group_files_by_schema_groups_by_exact_match(tmp_path):
    # a and b share a schema; c has an extra column -> its own group, even though
    # all three "look" related (same naming, similar content).
    pl.DataFrame({"episode_id": ["e1"], "setpoint_pos_0": [1.0]}).write_parquet(
        tmp_path / "a.parquet"
    )
    pl.DataFrame({"episode_id": ["e1"], "setpoint_pos_0": [2.0]}).write_parquet(
        tmp_path / "b.parquet"
    )
    pl.DataFrame({"episode_id": ["e1"], "setpoint_pos_0": [3.0], "extra_col": [1]}).write_parquet(
        tmp_path / "c.parquet"
    )

    groups = group_files_by_schema(
        files=["a.parquet", "b.parquet", "c.parquet"],
        source_url_fn=lambda f: str(tmp_path / f),
    )

    sizes = sorted(len(g.files) for g in groups.values())
    assert sizes == [1, 2]
    matching_group = next(g for g in groups.values() if len(g.files) == 2)
    assert matching_group.files == ["a.parquet", "b.parquet"]


def test_build_incremental_pipeline_ingests_group_end_to_end(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    pl.DataFrame(
        {
            "episode_id": ["e1", "e2"],
            "time_s": [0.0, 1.0],
            "setpoint_pos_0": [1.0, 2.0],
            "ctx_material": ["steel", "steel"],  # not a documented channel/metadata column
        }
    ).write_parquet(src_dir / "a.parquet")

    schema = pl.scan_parquet(src_dir / "a.parquet").collect_schema()
    group = SchemaGroup(schema=schema, files=["a.parquet"])
    target = str(tmp_path / "target")

    pipeline = build_incremental_pipeline(
        bucket="unused",
        group_id="schema_00",
        group=group,
        target=target,
        source_url_fn=lambda f: str(src_dir / f),
    )

    processed = pipeline.run()
    assert len(processed) == 1

    result = pl.read_delta(target)
    assert set(result.columns) == {"episode_id", "time_s", "setpoint_pos_0", "source_file"}
    assert result["source_file"].unique().to_list() == ["a.parquet"]
    assert result.sort("episode_id")["episode_id"].to_list() == ["a.parquet/e1", "a.parquet/e2"]

    # Rerun: watermark means nothing new to ingest.
    assert pipeline.run() == []


def test_build_incremental_pipeline_keeps_requested_extra_columns(tmp_path):
    # ctx_action isn't in the generic METADATA_COLUMNS set, so without
    # extra_columns it would be silently dropped (on_extra="drop").
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    pl.DataFrame(
        {
            "episode_id": ["e1"],
            "setpoint_pos_0": [1.0],
            "ctx_action": [2],
            "ctx_unrequested": ["dropped"],
        }
    ).write_parquet(src_dir / "a.parquet")

    schema = pl.scan_parquet(src_dir / "a.parquet").collect_schema()
    group = SchemaGroup(schema=schema, files=["a.parquet"])
    target = str(tmp_path / "target")

    pipeline = build_incremental_pipeline(
        bucket="unused",
        group_id="schema_00",
        group=group,
        target=target,
        source_url_fn=lambda f: str(src_dir / f),
        extra_columns=("ctx_action", "ctx_not_in_schema"),
    )
    pipeline.run()

    result = pl.read_delta(target)
    assert "ctx_action" in result.columns
    assert result["ctx_action"].to_list() == [2]
    assert "ctx_unrequested" not in result.columns
    assert "ctx_not_in_schema" not in result.columns


def test_build_incremental_pipeline_defaults_target_from_group_id(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    pl.DataFrame({"setpoint_pos_0": [1.0]}).write_parquet(src_dir / "a.parquet")
    schema = pl.scan_parquet(src_dir / "a.parquet").collect_schema()
    group = SchemaGroup(schema=schema, files=["a.parquet"])

    pipeline = build_incremental_pipeline(
        bucket="my-bucket",
        group_id="schema_03",
        group=group,
        source_url_fn=lambda f: str(src_dir / f),
    )

    assert pipeline.target == "s3://my-bucket/delta/factorynet_schema_03"


def test_stream_factorynet_to_bucket_runs_one_pipeline_per_group(monkeypatch):
    calls = []

    class _FakePipeline:
        def __init__(self, group_id):
            self._group_id = group_id

        def run(self):
            return [f"{self._group_id}-file"]

    def _fake_build(bucket, group_id, group, target=None, storage_options=None, source_url_fn=None):
        calls.append((bucket, group_id))
        return _FakePipeline(group_id)

    monkeypatch.setattr(factorynet_module, "build_incremental_pipeline", _fake_build)

    groups = {
        "schema_00": SchemaGroup(schema=pl.Schema({"x": pl.Int64}), files=["a.parquet"]),
        "schema_01": SchemaGroup(schema=pl.Schema({"y": pl.Int64}), files=["b.parquet"]),
    }

    result = stream_factorynet_to_bucket("my-bucket", groups=groups)

    assert result == {"schema_00": ["schema_00-file"], "schema_01": ["schema_01-file"]}
    assert calls == [("my-bucket", "schema_00"), ("my-bucket", "schema_01")]
