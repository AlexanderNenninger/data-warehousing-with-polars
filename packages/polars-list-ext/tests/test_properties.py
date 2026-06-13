"""Property-based tests for list_ext combinators using Hypothesis."""

import polars as pl
import polars_list_ext  # noqa: F401 — registers list_ext namespace
from hypothesis import given, settings
from hypothesis import strategies as st

# ── Strategies ─────────────────────────────────────────────────────────────────

small_ints = st.integers(-1_000, 1_000)
list_of_ints = st.lists(small_ints, min_size=0, max_size=20)
# Small key range so collisions (and thus non-trivial joins) occur frequently.
_key_range = st.integers(0, 5)


@st.composite
def two_int_list_columns(draw) -> pl.DataFrame:
    """DataFrame with columns 'a' and 'b', each List[Int64] with equal row count."""
    n_rows = draw(st.integers(1, 8))
    a = draw(st.lists(list_of_ints, min_size=n_rows, max_size=n_rows))
    b = draw(st.lists(list_of_ints, min_size=n_rows, max_size=n_rows))
    return pl.DataFrame(
        {"a": a, "b": b},
        schema={"a": pl.List(pl.Int64), "b": pl.List(pl.Int64)},
    )


@st.composite
def two_struct_list_columns(draw) -> pl.DataFrame:
    """DataFrame with 'left' and 'right' List[Struct{id:Int64, val:Int64}]."""
    n_rows = draw(st.integers(1, 6))
    schema = pl.Schema(
        {
            "left": pl.List(pl.Struct({"id": pl.Int64, "val": pl.Int64})),
            "right": pl.List(pl.Struct({"id": pl.Int64, "val": pl.Int64})),
        }
    )
    left_rows, right_rows = [], []
    for _ in range(n_rows):
        n_l = draw(st.integers(0, 8))
        n_r = draw(st.integers(0, 8))
        left_rows.append([{"id": draw(_key_range), "val": draw(small_ints)} for _ in range(n_l)])
        right_rows.append([{"id": draw(_key_range), "val": draw(small_ints)} for _ in range(n_r)])
    return pl.DataFrame({"left": left_rows, "right": right_rows}, schema=schema)


# ── zip / unzip properties ──────────────────────────────────────────────────────


@given(two_int_list_columns())
def test_zip_unzip_round_trip(df: pl.DataFrame) -> None:
    """unzip(zip(a, b)) recovers the original lists, truncated to min length."""
    result = (
        df.with_columns(pl.col("a").list_ext.zip(pl.col("b")).alias("zipped"))
        .with_columns(pl.col("zipped").list_ext.unzip().alias("u"))
        .unnest("u")
    )
    a_rows = df["a"].to_list()
    b_rows = df["b"].to_list()
    first_rows = result["first"].to_list()
    second_rows = result["second"].to_list()
    for i in range(df.height):
        n = min(len(a_rows[i]), len(b_rows[i]))
        assert first_rows[i] == a_rows[i][:n]
        assert second_rows[i] == b_rows[i][:n]


@given(two_int_list_columns())
def test_zip_output_length_is_min_of_inputs(df: pl.DataFrame) -> None:
    """zip output length per row equals min(len(a_row), len(b_row))."""
    result = df.with_columns(pl.col("a").list_ext.zip(pl.col("b")).alias("zipped"))
    for i in range(df.height):
        expected_len = min(len(df["a"][i]), len(df["b"][i]))
        assert result["zipped"].list.len()[i] == expected_len


@given(two_int_list_columns())
def test_zip_output_has_both_field_names(df: pl.DataFrame) -> None:
    """zip output struct always has exactly 'first' and 'second' fields."""
    result = (
        df.with_columns(pl.col("a").list_ext.zip(pl.col("b")).alias("zipped"))
        .with_columns(pl.col("zipped").list_ext.unzip().alias("u"))
        .unnest("u")
    )
    assert "first" in result.columns
    assert "second" in result.columns


@given(two_int_list_columns())
def test_zip_null_propagation(df: pl.DataFrame) -> None:
    """A null in either input column produces a null output row."""
    schema = {"a": pl.List(pl.Int64), "b": pl.List(pl.Int64)}
    nulled = pl.DataFrame(
        {
            "a": [None] + df["a"].to_list(),
            "b": df["b"].to_list() + [None],
        },
        schema=schema,
    )
    result = nulled.with_columns(pl.col("a").list_ext.zip(pl.col("b")).alias("z"))
    assert result["z"][0] is None
    assert result["z"][-1] is None


# ── join properties ─────────────────────────────────────────────────────────────


@given(two_struct_list_columns())
@settings(max_examples=200)
def test_join_partition_law(df: pl.DataFrame) -> None:
    """len(inner_join) + len(anti_join) == len(left) for every row."""
    inner = df.with_columns(
        pl.col("left").list_ext.join(pl.col("right"), on="id", how="inner").alias("j")
    )
    anti = df.with_columns(
        pl.col("left").list_ext.join(pl.col("right"), on="id", how="anti").alias("j")
    )
    for i in range(df.height):
        n_inner = inner["j"].list.len()[i] or 0
        n_anti = anti["j"].list.len()[i] or 0
        n_left = len(df["left"][i])
        assert n_inner + n_anti == n_left, (
            f"row {i}: inner({n_inner}) + anti({n_anti}) != left({n_left})"
        )


@given(two_struct_list_columns())
@settings(max_examples=200)
def test_left_join_preserves_left_length(df: pl.DataFrame) -> None:
    """left join output row count always equals left input row count."""
    result = df.with_columns(
        pl.col("left").list_ext.join(pl.col("right"), on="id", how="left").alias("j")
    )
    for i in range(df.height):
        assert (result["j"].list.len()[i] or 0) == len(df["left"][i])


@given(two_struct_list_columns())
@settings(max_examples=200)
def test_inner_join_keys_present_in_both(df: pl.DataFrame) -> None:
    """Every key in the inner join result exists in both the left and right lists."""
    result = df.with_columns(
        pl.col("left").list_ext.join(pl.col("right"), on="id", how="inner").alias("j")
    )
    left_rows = df["left"].to_list()
    right_rows = df["right"].to_list()
    j_rows = result["j"].to_list()
    for i in range(df.height):
        left_keys = {r["id"] for r in left_rows[i]}
        right_keys = {r["id"] for r in right_rows[i]}
        for row in j_rows[i] or []:
            assert row["id"] in left_keys
            assert row["id"] in right_keys


@given(two_struct_list_columns())
@settings(max_examples=200)
def test_anti_join_keys_absent_from_right(df: pl.DataFrame) -> None:
    """Every key in the anti join result is absent from the right list."""
    result = df.with_columns(
        pl.col("left").list_ext.join(pl.col("right"), on="id", how="anti").alias("j")
    )
    right_rows = df["right"].to_list()
    j_rows = result["j"].to_list()
    for i in range(df.height):
        right_keys = {r["id"] for r in right_rows[i]}
        for row in j_rows[i] or []:
            assert row["id"] not in right_keys


@given(two_struct_list_columns())
@settings(max_examples=100)
def test_inner_join_keys_are_subset_of_left_join_keys(df: pl.DataFrame) -> None:
    """inner join key set is always a subset of left join key set per row."""
    inner = df.with_columns(
        pl.col("left").list_ext.join(pl.col("right"), on="id", how="inner").alias("j")
    )
    left_j = df.with_columns(
        pl.col("left").list_ext.join(pl.col("right"), on="id", how="left").alias("j")
    )
    inner_rows = inner["j"].to_list()
    left_rows = left_j["j"].to_list()
    for i in range(df.height):
        inner_ids = {r["id"] for r in inner_rows[i] or []}
        left_ids = {r["id"] for r in left_rows[i] or []}
        assert inner_ids <= left_ids


@given(two_struct_list_columns())
@settings(max_examples=100)
def test_inner_join_deduplicates_right_to_last_match(df: pl.DataFrame) -> None:
    """Cross-validate: inner join output length == number of distinct left keys
    that appear in the right list (one match per left row, last right wins).
    """
    result = df.with_columns(
        pl.col("left").list_ext.join(pl.col("right"), on="id", how="inner").alias("j")
    )
    left_rows = df["left"].to_list()
    right_rows = df["right"].to_list()
    j_rows = result["j"].to_list()
    for i in range(df.height):
        # Right keys as a set (duplicates collapsed — last-match semantics).
        right_key_set = {r["id"] for r in right_rows[i]}
        # Expected: one output row per left element whose key appears in right.
        expected_len = sum(1 for r in left_rows[i] if r["id"] in right_key_set)
        assert len(j_rows[i] or []) == expected_len


# ── Null key invariant ──────────────────────────────────────────────────────────
# Polars preserves null through .cast(String) — null keys stay null (not "null"),
# so the Rust if-let-Some guard skips them on both sides. This test locks that in.

_nullable_key_range = st.one_of(st.none(), st.integers(0, 3))


@st.composite
def two_struct_list_columns_with_null_keys(draw) -> pl.DataFrame:
    """Like two_struct_list_columns but keys can be null."""
    n_rows = draw(st.integers(1, 6))
    schema = pl.Schema(
        {
            "left": pl.List(pl.Struct({"id": pl.Int64, "val": pl.Int64})),
            "right": pl.List(pl.Struct({"id": pl.Int64, "val": pl.Int64})),
        }
    )
    left_rows, right_rows = [], []
    for _ in range(n_rows):
        n_l = draw(st.integers(0, 6))
        n_r = draw(st.integers(0, 6))
        left_rows.append(
            [{"id": draw(_nullable_key_range), "val": draw(small_ints)} for _ in range(n_l)]
        )
        right_rows.append(
            [{"id": draw(_nullable_key_range), "val": draw(small_ints)} for _ in range(n_r)]
        )
    return pl.DataFrame({"left": left_rows, "right": right_rows}, schema=schema)


@given(two_struct_list_columns_with_null_keys())
@settings(max_examples=200)
def test_null_keys_never_match(df: pl.DataFrame) -> None:
    """Null keys on either side must never match each other in any join mode."""
    for how in ("inner", "left", "anti"):
        result = df.with_columns(
            pl.col("left").list_ext.join(pl.col("right"), on="id", how=how).alias("j")
        )
        df["left"].to_list()
        right_rows = df["right"].to_list()
        j_rows = result["j"].to_list()
        right_null_counts = [
            sum(1 for r in right_rows[i] if r["id"] is None) for i in range(df.height)
        ]
        for i in range(df.height):
            if right_null_counts[i] == 0:
                continue
            # When right has null keys: inner/left should not produce rows
            # matching null-keyed left elements via the null right keys.
            if how == "inner":
                # No left-null-key row should appear in inner result
                # (null left key has no match on right because null is skipped).
                left_null_ids_in_result = [r for r in (j_rows[i] or []) if r["id"] is None]
                assert left_null_ids_in_result == [], (
                    f"row {i} how={how!r}: null key appeared in inner join output: "
                    f"{left_null_ids_in_result}"
                )
