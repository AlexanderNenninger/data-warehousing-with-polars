"""Tests for SCD Type 2 and Type 4 write semantics."""

from datetime import date
from pathlib import Path

import polars as pl
from data_warehousing_with_polars.scd import (
    _partition_predicate,
    _sink_scd2,
    _sink_scd4,
    _sql_literal,
    _upsert_overwrite,
)
from deltalake import DeltaTable, write_deltalake

# ── _sink_scd2 ────────────────────────────────────────────────────────────────


def test_scd2_first_write_creates_table_with_bookkeeping_columns(tmp_path):
    target = str(tmp_path / "scd2")
    lf = pl.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]}).lazy()

    _sink_scd2(target, lf, merge_on="id")

    result = pl.scan_delta(target).collect()
    assert len(result) == 2
    assert "valid_from" in result.columns
    assert "valid_to" in result.columns
    assert "is_current" in result.columns


def test_scd2_first_write_all_rows_are_current(tmp_path):
    target = str(tmp_path / "scd2")
    lf = pl.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]}).lazy()

    _sink_scd2(target, lf, merge_on="id")

    result = pl.scan_delta(target).collect()
    assert result["is_current"].all()
    assert result["valid_to"].is_null().all()


def test_scd2_update_closes_old_version_and_appends_new(tmp_path):
    target = str(tmp_path / "scd2")
    _sink_scd2(target, pl.DataFrame({"id": [1], "name": ["Alice"]}).lazy(), merge_on="id")
    _sink_scd2(target, pl.DataFrame({"id": [1], "name": ["Alice V2"]}).lazy(), merge_on="id")

    result = pl.scan_delta(target).collect()
    current = result.filter(pl.col("is_current"))
    historical = result.filter(~pl.col("is_current"))

    assert len(current) == 1
    assert current["name"][0] == "Alice V2"
    assert len(historical) == 1
    assert historical["name"][0] == "Alice"
    assert historical["valid_to"][0] is not None


def test_scd2_new_key_appended_without_closing_others(tmp_path):
    target = str(tmp_path / "scd2")
    _sink_scd2(target, pl.DataFrame({"id": [1], "name": ["Alice"]}).lazy(), merge_on="id")
    _sink_scd2(target, pl.DataFrame({"id": [2], "name": ["Bob"]}).lazy(), merge_on="id")

    result = pl.scan_delta(target).collect()
    assert len(result) == 2
    assert result["is_current"].all()


def test_scd2_idempotent_rerun_does_not_duplicate(tmp_path):
    """Re-running with the same batch must not add duplicate current versions."""
    target = str(tmp_path / "scd2")
    _sink_scd2(target, pl.DataFrame({"id": [1], "name": ["Alice"]}).lazy(), merge_on="id")

    # Simulate a re-run by reading the existing valid_from and injecting it,
    # then calling again — the dedup on (id, valid_from) should prevent a new row.
    existing_vf = pl.scan_delta(target).select("valid_from").collect()["valid_from"][0]
    (
        pl.DataFrame({"id": [1], "name": ["Alice"]})
        .lazy()
        .with_columns(pl.lit(existing_vf).alias("_note"))
    )
    # Call scd2 with a genuinely new timestamp; re-run detection is key-based
    # so a second call with the same key + same timestamp is deduplicated.
    # We just verify that calling twice with the same payload produces only 1 current row.
    _sink_scd2(target, pl.DataFrame({"id": [1], "name": ["Alice"]}).lazy(), merge_on="id")

    result = pl.scan_delta(target).collect()
    current = result.filter(pl.col("is_current"))
    assert len(current) == 1


def test_scd2_partition_by(tmp_path):
    target = str(tmp_path / "scd2_partitioned")
    lf = pl.DataFrame({"id": [1, 2], "region": ["EU", "US"], "name": ["A", "B"]}).lazy()

    _sink_scd2(target, lf, merge_on="id", partition_by="region")

    result = pl.scan_delta(target).collect()
    assert len(result) == 2


# ── _sink_scd4 ────────────────────────────────────────────────────────────────


def test_scd4_first_write_creates_main_table(tmp_path):
    target = str(tmp_path / "scd4_main")
    history = str(tmp_path / "scd4_hist")
    lf = pl.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]}).lazy()

    _sink_scd4(target, history, lf, merge_on="id")

    result = pl.scan_delta(target).collect()
    assert len(result) == 2
    assert list(sorted(result["id"].to_list())) == [1, 2]


def test_scd4_first_write_does_not_create_history_table(tmp_path):
    """No existing records means nothing to archive on first write."""
    target = str(tmp_path / "scd4_main")
    history = str(tmp_path / "scd4_hist")
    lf = pl.DataFrame({"id": [1], "name": ["Alice"]}).lazy()

    _sink_scd4(target, history, lf, merge_on="id")

    assert not Path(history).exists()


def test_scd4_update_archives_superseded_record(tmp_path):
    target = str(tmp_path / "scd4_main")
    history = str(tmp_path / "scd4_hist")

    _sink_scd4(target, history, pl.DataFrame({"id": [1], "name": ["Alice"]}).lazy(), merge_on="id")
    _sink_scd4(
        target, history, pl.DataFrame({"id": [1], "name": ["Alice V2"]}).lazy(), merge_on="id"
    )

    main = pl.scan_delta(target).collect()
    hist = pl.scan_delta(history).collect()

    assert len(main) == 1
    assert main["name"][0] == "Alice V2"
    assert len(hist) == 1
    assert hist["name"][0] == "Alice"
    assert "superseded_at" in hist.columns


def test_scd4_history_accumulates_across_updates(tmp_path):
    target = str(tmp_path / "scd4_main")
    history = str(tmp_path / "scd4_hist")

    _sink_scd4(target, history, pl.DataFrame({"id": [1], "name": ["v1"]}).lazy(), merge_on="id")
    _sink_scd4(target, history, pl.DataFrame({"id": [1], "name": ["v2"]}).lazy(), merge_on="id")
    _sink_scd4(target, history, pl.DataFrame({"id": [1], "name": ["v3"]}).lazy(), merge_on="id")

    hist = pl.scan_delta(history).collect()
    assert len(hist) == 2  # v1 and v2 archived; v3 is current

    main = pl.scan_delta(target).collect()
    assert main["name"][0] == "v3"


def test_scd4_unrelated_key_not_archived(tmp_path):
    target = str(tmp_path / "scd4_main")
    history = str(tmp_path / "scd4_hist")

    _sink_scd4(
        target,
        history,
        pl.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]}).lazy(),
        merge_on="id",
    )
    # Only update id=1; id=2 should remain in main untouched with no history entry.
    _sink_scd4(
        target,
        history,
        pl.DataFrame({"id": [1], "name": ["Alice V2"]}).lazy(),
        merge_on="id",
    )

    main = pl.scan_delta(target).collect()
    hist = pl.scan_delta(history).collect()

    assert len(main) == 2
    assert len(hist) == 1
    assert hist["id"][0] == 1


def test_scd4_partition_by(tmp_path):
    target = str(tmp_path / "scd4_main")
    history = str(tmp_path / "scd4_hist")
    lf = pl.DataFrame({"id": [1, 2], "region": ["EU", "US"], "name": ["A", "B"]}).lazy()

    _sink_scd4(target, history, lf, merge_on="id", partition_by="region")

    result = pl.scan_delta(target).collect()
    assert len(result) == 2


# ── replaceWhere predicate helpers ────────────────────────────────────────────


def test_sql_literal_integer_is_cast_to_column_width():
    # delta-rs rejects Int32-vs-Int64 comparisons, so integers carry an explicit cast.
    assert _sql_literal(2024, pl.Int32()) == "CAST(2024 AS INT)"
    assert _sql_literal(2024, pl.Int64()) == "CAST(2024 AS BIGINT)"


def test_sql_literal_string_is_quoted_and_escaped():
    assert _sql_literal("EU", pl.Utf8()) == "'EU'"
    assert _sql_literal("O'Hare", pl.Utf8()) == "'O''Hare'"


def test_sql_literal_date_and_bool():
    assert _sql_literal(date(2024, 1, 1), pl.Date()) == "CAST('2024-01-01' AS DATE)"
    assert _sql_literal(True, pl.Boolean()) == "true"


def test_partition_predicate_multi_column():
    df = pl.DataFrame({"year": pl.Series([2024, 2025], dtype=pl.Int32), "region": ["EU", "US"]})
    pred = _partition_predicate(df, ["year", "region"])
    assert pred == "year IN (CAST(2024 AS INT), CAST(2025 AS INT)) AND region IN ('EU', 'US')"


def test_partition_predicate_falls_back_on_null_partition_value():
    df = pl.DataFrame({"region": ["EU", None]})
    assert _partition_predicate(df, ["region"]) is None


# ── _upsert_overwrite ─────────────────────────────────────────────────────────


def _files_by_partition(target: str, part_col: str) -> dict[str, set[str]]:
    import re

    out: dict[str, set[str]] = {}
    for uri in DeltaTable(target).file_uris():
        m = re.search(rf"{part_col}=([^/]+)", uri)
        out.setdefault(m.group(1) if m else "?", set()).add(uri)
    return out


def test_upsert_overwrite_partitioned_rewrites_only_touched_partitions(tmp_path):
    target = str(tmp_path / "t")
    df = pl.DataFrame(
        {
            "id": list(range(6)),
            "year": pl.Series([2020, 2020, 2021, 2021, 2022, 2022], dtype=pl.Int32),
            "v": [1.0] * 6,
        }
    )
    write_deltalake(target, df.to_arrow(), mode="overwrite", partition_by=["year"])
    before = _files_by_partition(target, "year")

    # Update only year 2021 (and insert a new id there).
    batch = pl.DataFrame(
        {"id": [2, 3, 99], "year": pl.Series([2021, 2021, 2021], dtype=pl.Int32), "v": [9.0] * 3}
    )
    _upsert_overwrite(target, batch, ["id"], ["year"])
    after = _files_by_partition(target, "year")

    # Untouched partitions keep their exact files; only 2021 is rewritten.
    assert after["2020"] == before["2020"]
    assert after["2022"] == before["2022"]
    assert after["2021"] != before["2021"]

    out = pl.scan_delta(target).collect().sort("id")
    assert len(out) == 7  # 6 + 1 new
    assert out.filter(pl.col("year") == 2021)["v"].unique().to_list() == [9.0]
    assert out.filter(pl.col("year") == 2020)["v"].unique().to_list() == [1.0]
    assert out.group_by("id").len().filter(pl.col("len") > 1).is_empty()


def test_upsert_overwrite_unpartitioned_full_rewrite(tmp_path):
    target = str(tmp_path / "t")
    df = pl.DataFrame({"id": [1, 2, 3], "v": [1.0, 1.0, 1.0]})
    write_deltalake(target, df.to_arrow(), mode="overwrite")

    batch = pl.DataFrame({"id": [2, 4], "v": [9.0, 9.0]})
    _upsert_overwrite(target, batch, ["id"], None)

    out = pl.scan_delta(target).collect().sort("id")
    assert out["id"].to_list() == [1, 2, 3, 4]
    assert out.filter(pl.col("id") == 2)["v"][0] == 9.0
    assert out.group_by("id").len().filter(pl.col("len") > 1).is_empty()
