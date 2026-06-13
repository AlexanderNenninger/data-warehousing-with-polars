"""Tests for maintenance module: run counter and table health operations."""

import polars as pl
import pytest
from data_warehousing_with_polars.maintenance import (
    _load_run_count,
    _save_run_count,
    maintain,
)
from deltalake import write_deltalake

# ── Run counter ───────────────────────────────────────────────────────────────


def test_load_run_count_returns_zero_when_no_table(tmp_path):
    count = _load_run_count(str(tmp_path / "watermark"))
    assert count == 0


def test_save_and_load_run_count_round_trip(tmp_path):
    store = str(tmp_path / "watermark")
    _save_run_count(store, 1)
    assert _load_run_count(store) == 1


def test_save_run_count_multiple_times_returns_latest(tmp_path):
    store = str(tmp_path / "watermark")
    _save_run_count(store, 1)
    _save_run_count(store, 2)
    _save_run_count(store, 3)
    assert _load_run_count(store) == 3


def test_save_run_count_stores_at_runs_suffix(tmp_path):
    store = str(tmp_path / "watermark")
    _save_run_count(store, 5)
    # Verify the sub-table path exists
    import pathlib

    runs_path = pathlib.Path(store + "/_runs")
    assert runs_path.exists()


# ── maintain ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def delta_table(tmp_path):
    """Create a small Delta table for maintenance tests."""
    path = str(tmp_path / "delta")
    df = pl.DataFrame({"id": [1, 2, 3, 4, 5], "val": ["a", "b", "c", "d", "e"]})
    write_deltalake(path, df.to_arrow(), mode="overwrite")
    return path


def test_maintain_compact_runs_without_error(delta_table):
    maintain(delta_table, compact=True, vacuum=False)


def test_maintain_vacuum_runs_without_error(delta_table):
    maintain(delta_table, compact=False, vacuum=True, retention_hours=168)


def test_maintain_compact_and_vacuum(delta_table):
    maintain(delta_table, compact=True, vacuum=True, retention_hours=168)


def test_maintain_z_order_runs_without_error(delta_table):
    maintain(delta_table, z_order_by="id", vacuum=False)


def test_maintain_z_order_list(delta_table):
    # Write a table with two orderable columns
    import pathlib

    from deltalake import write_deltalake as wdl

    path = str(pathlib.Path(delta_table).parent / "delta2")
    df = pl.DataFrame({"id": [1, 2, 3], "score": [10, 20, 30], "label": ["x", "y", "z"]})
    wdl(path, df.to_arrow(), mode="overwrite")
    maintain(path, z_order_by=["id", "score"], vacuum=False)


def test_maintain_table_not_found_raises(tmp_path):
    from deltalake.exceptions import TableNotFoundError

    with pytest.raises(TableNotFoundError):
        maintain(str(tmp_path / "nonexistent"), compact=True, vacuum=False)


def test_maintain_no_ops_does_not_raise(delta_table):
    maintain(delta_table, compact=False, vacuum=False)
