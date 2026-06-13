"""Smoke tests: confirm the compiled plugin loads and core expressions run."""

import polars as pl
import polars_list_ext as ple


def test_version_is_exposed():
    assert isinstance(ple._internal.__version__, str)


def test_apply_fft_returns_expected_bin_count():
    # A real FFT of a length-N signal yields N // 2 + 1 magnitude bins.
    n = 16
    df = pl.DataFrame({"sig": [[float(i % 4) for i in range(n)]]})
    out = df.with_columns(ple.apply_fft("sig", sample_rate=n).alias("fft"))
    assert out["fft"].list.len().to_list() == [n // 2 + 1]


def test_apply_fft_is_lazy_compatible():
    n = 8
    lf = pl.LazyFrame({"sig": [[1.0] * n, [0.0] * n]})
    out = lf.with_columns(ple.apply_fft("sig", sample_rate=n).alias("fft")).collect()
    assert out.height == 2
    assert out["fft"].list.len().to_list() == [n // 2 + 1, n // 2 + 1]


# ── list_ext namespace: zip / unzip ──────────────────────────────────────────


def test_zip_basic():
    df = pl.DataFrame({"a": [[1, 2, 3]], "b": [[4, 5, 6]]})
    out = df.with_columns(pl.col("a").list_ext.zip(pl.col("b")).alias("zipped"))
    zipped = out["zipped"]
    assert zipped.list.len().to_list() == [3]
    first_row = zipped[0]
    assert first_row[0]["first"] == 1
    assert first_row[0]["second"] == 4
    assert first_row[2]["first"] == 3
    assert first_row[2]["second"] == 6


def test_zip_unzip_round_trip():
    df = pl.DataFrame({"a": [[1, 2, 3], [10, 20]], "b": [[4, 5, 6], [40, 50]]})
    out = (
        df.with_columns(pl.col("a").list_ext.zip(pl.col("b")).alias("zipped"))
        .with_columns(pl.col("zipped").list_ext.unzip().alias("unzipped"))
        .unnest("unzipped")
    )
    assert out["first"].to_list() == [[1, 2, 3], [10, 20]]
    assert out["second"].to_list() == [[4, 5, 6], [40, 50]]


def test_zip_mismatched_lengths():
    df = pl.DataFrame({"a": [[1, 2, 3]], "b": [[10, 20]]})
    out = df.with_columns(pl.col("a").list_ext.zip(pl.col("b")).alias("zipped"))
    # Shorter list wins — length should be 2.
    assert out["zipped"].list.len().to_list() == [2]


def test_zip_null_row_propagates():
    df = pl.DataFrame(
        {"a": [None, [1, 2]], "b": [[3, 4], [5, 6]]},
        schema={"a": pl.List(pl.Int64), "b": pl.List(pl.Int64)},
    )
    out = df.with_columns(pl.col("a").list_ext.zip(pl.col("b")).alias("zipped"))
    assert out["zipped"][0] is None
    assert out["zipped"][1] is not None


def test_unzip_three_fields():
    schema = pl.Schema(
        {"triples": pl.List(pl.Struct({"x": pl.Int64, "y": pl.Int64, "z": pl.Int64}))}
    )
    df = pl.DataFrame(
        {"triples": [[{"x": 1, "y": 2, "z": 3}, {"x": 4, "y": 5, "z": 6}]]},
        schema=schema,
    )
    out = df.with_columns(pl.col("triples").list_ext.unzip().alias("u")).unnest("u")
    assert out["x"].to_list() == [[1, 4]]
    assert out["y"].to_list() == [[2, 5]]
    assert out["z"].to_list() == [[3, 6]]


# ── list_ext namespace: join ─────────────────────────────────────────────────

_ORDER_SCHEMA = pl.Schema({"o": pl.List(pl.Struct({"id": pl.Int64, "qty": pl.Int64}))})
_PRODUCT_SCHEMA = pl.Schema({"p": pl.List(pl.Struct({"id": pl.Int64, "name": pl.String}))})


def test_join_inner_basic():
    df = pl.DataFrame(
        {
            "o": [[{"id": 1, "qty": 10}, {"id": 2, "qty": 5}, {"id": 3, "qty": 2}]],
            "p": [[{"id": 1, "name": "A"}, {"id": 3, "name": "C"}]],
        },
        schema={**_ORDER_SCHEMA, **_PRODUCT_SCHEMA},
    )
    out = df.with_columns(pl.col("o").list_ext.join(pl.col("p"), on="id").alias("j"))
    # Only ids 1 and 3 match.
    j = out["j"]
    assert j.list.len().to_list() == [2]
    ids = [row["id"] for row in j[0]]
    assert ids == [1, 3]
    names = [row["name"] for row in j[0]]
    assert names == ["A", "C"]


def test_join_left_keeps_all_left_rows():
    df = pl.DataFrame(
        {
            "o": [[{"id": 1, "qty": 10}, {"id": 99, "qty": 5}]],
            "p": [[{"id": 1, "name": "A"}]],
        },
        schema={**_ORDER_SCHEMA, **_PRODUCT_SCHEMA},
    )
    out = df.with_columns(pl.col("o").list_ext.join(pl.col("p"), on="id", how="left").alias("j"))
    j = out["j"]
    assert j.list.len().to_list() == [2]  # both left rows preserved
    names = [row["name"] for row in j[0]]
    assert names[0] == "A"
    assert names[1] is None  # no match for id=99


def test_join_anti_returns_unmatched():
    df = pl.DataFrame(
        {
            "o": [[{"id": 1, "qty": 10}, {"id": 99, "qty": 5}]],
            "p": [[{"id": 1, "name": "A"}]],
        },
        schema={**_ORDER_SCHEMA, **_PRODUCT_SCHEMA},
    )
    out = df.with_columns(pl.col("o").list_ext.join(pl.col("p"), on="id", how="anti").alias("j"))
    j = out["j"]
    assert j.list.len().to_list() == [1]  # only id=99
    assert j[0][0]["id"] == 99


def test_join_suffix_on_name_collision():
    schema_a = pl.Schema({"a": pl.List(pl.Struct({"id": pl.Int64, "val": pl.Int64}))})
    schema_b = pl.Schema({"b": pl.List(pl.Struct({"id": pl.Int64, "val": pl.String}))})
    df = pl.DataFrame(
        {"a": [[{"id": 1, "val": 42}]], "b": [[{"id": 1, "val": "x"}]]},
        schema={**schema_a, **schema_b},
    )
    out = df.with_columns(pl.col("a").list_ext.join(pl.col("b"), on="id", suffix="_b").alias("j"))
    j = out["j"]
    row = j[0][0]
    assert row["val"] == 42  # left val
    assert row["val_b"] == "x"  # right val with suffix


# ── list_ext namespace: enumerate / dedup / rotate / windows / chunks / position / flat_map ──


def test_enumerate_basic():
    df = pl.DataFrame({"xs": [["a", "b", "c"]]})
    out = df.with_columns(pl.col("xs").list_ext.enumerate().alias("e"))
    rows = out["e"].to_list()
    assert rows[0][0] == {"index": 0, "value": "a"}
    assert rows[0][2] == {"index": 2, "value": "c"}


def test_dedup_removes_consecutive_duplicates():
    df = pl.DataFrame({"xs": [[1, 1, 2, 3, 3, 3, 2, 2]]}, schema={"xs": pl.List(pl.Int64)})
    out = df.with_columns(pl.col("xs").list_ext.dedup().alias("d"))
    assert out["d"].to_list() == [[1, 2, 3, 2]]


def test_dedup_keeps_non_adjacent_duplicates():
    df = pl.DataFrame({"xs": [[1, 2, 1, 2]]}, schema={"xs": pl.List(pl.Int64)})
    out = df.with_columns(pl.col("xs").list_ext.dedup().alias("d"))
    assert out["d"].to_list() == [[1, 2, 1, 2]]


def test_rotate_right():
    df = pl.DataFrame({"xs": [[1, 2, 3, 4, 5]]}, schema={"xs": pl.List(pl.Int64)})
    out = df.with_columns(pl.col("xs").list_ext.rotate(2).alias("r"))
    assert out["r"].to_list() == [[4, 5, 1, 2, 3]]


def test_rotate_left():
    df = pl.DataFrame({"xs": [[1, 2, 3, 4, 5]]}, schema={"xs": pl.List(pl.Int64)})
    out = df.with_columns(pl.col("xs").list_ext.rotate(-2).alias("r"))
    assert out["r"].to_list() == [[3, 4, 5, 1, 2]]


def test_windows_basic():
    df = pl.DataFrame({"xs": [[1, 2, 3, 4, 5]]}, schema={"xs": pl.List(pl.Int64)})
    out = df.with_columns(pl.col("xs").list_ext.windows(3).alias("w"))
    assert out["w"].to_list() == [[[1, 2, 3], [2, 3, 4], [3, 4, 5]]]


def test_windows_with_step():
    df = pl.DataFrame({"xs": [[1, 2, 3, 4, 5, 6]]}, schema={"xs": pl.List(pl.Int64)})
    out = df.with_columns(pl.col("xs").list_ext.windows(3, step=2).alias("w"))
    assert out["w"].to_list() == [[[1, 2, 3], [3, 4, 5]]]


def test_windows_shorter_than_size_returns_empty():
    df = pl.DataFrame({"xs": [[1, 2]]}, schema={"xs": pl.List(pl.Int64)})
    out = df.with_columns(pl.col("xs").list_ext.windows(5).alias("w"))
    assert out["w"].list.len().to_list() == [0]


def test_chunks_basic():
    df = pl.DataFrame({"xs": [[1, 2, 3, 4, 5]]}, schema={"xs": pl.List(pl.Int64)})
    out = df.with_columns(pl.col("xs").list_ext.chunks(2).alias("c"))
    assert out["c"].to_list() == [[[1, 2], [3, 4], [5]]]


def test_chunks_exact_division():
    df = pl.DataFrame({"xs": [[1, 2, 3, 4]]}, schema={"xs": pl.List(pl.Int64)})
    out = df.with_columns(pl.col("xs").list_ext.chunks(2).alias("c"))
    assert out["c"].to_list() == [[[1, 2], [3, 4]]]


def test_position_finds_first_match():
    df = pl.DataFrame({"xs": [[1.0, 2.0, 3.0, 2.0]]})
    out = df.with_columns(pl.col("xs").list_ext.position("eq", 2.0).alias("p"))
    assert out["p"].to_list() == [1]


def test_position_returns_null_when_no_match():
    df = pl.DataFrame({"xs": [[1.0, 2.0, 3.0]]})
    out = df.with_columns(pl.col("xs").list_ext.position("gt", 10.0).alias("p"))
    assert out["p"].to_list() == [None]


def test_flat_map_mul():
    df = pl.DataFrame({"xs": [[1.0, 2.0, 3.0]]})
    out = df.with_columns(pl.col("xs").list_ext.flat_map("mul", 2.0).alias("m"))
    assert out["m"].to_list() == [[2.0, 4.0, 6.0]]
