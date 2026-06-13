"""Memory-boundedness tests for each @incremental use-case.

Each test spawns a fresh Python interpreter subprocess via ``subprocess.run``
so that Polars' rayon thread pool is never inherited from the parent pytest
process.  ``os.fork()``-based approaches (e.g. ``pytest-forked``) deadlock on
macOS because the child inherits the parent's already-initialised thread pool.

The actual pipeline setup, RSS measurement, and assertion live in
``_memory_impl.py``.  This file contains only thin dispatcher tests.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_IMPL = Path(__file__).parent / "_memory_impl.py"


def _run(name: str, tmp_path: Path) -> None:
    """Spawn a fresh interpreter to run ``run_<name>`` in ``_memory_impl.py``."""
    result = subprocess.run(
        [sys.executable, str(_IMPL), name, str(tmp_path)],
        capture_output=True,
        text=True,
    )
    output = (result.stdout + result.stderr).strip()
    if output:
        print(f"\n{output}")
    if result.returncode != 0:
        pytest.fail(f"Memory worker '{name}' failed:\n{output}")


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_memory_upsert(tmp_path: Path) -> None:
    """Basic upsert (merge_on) on a partitioned dataset stays within memory bound."""
    _run("upsert", tmp_path)


def test_memory_append_only(tmp_path: Path) -> None:
    """Append-only (no merge_on) on a partitioned dataset stays within memory bound."""
    _run("append_only", tmp_path)


def test_memory_fan_in(tmp_path: Path) -> None:
    """Fan-in from two partitioned source directories stays within memory bound."""
    _run("fan_in", tmp_path)


def test_memory_csv_format(tmp_path: Path) -> None:
    """CSV file_format on a partitioned dataset stays within memory bound."""
    _run("csv_format", tmp_path)


def test_memory_ndjson_format(tmp_path: Path) -> None:
    """NDJSON file_format on a partitioned dataset stays within memory bound."""
    _run("ndjson_format", tmp_path)


def test_memory_scd2(tmp_path: Path) -> None:
    """SCD Type 2 on a partitioned dataset stays within memory bound."""
    _run("scd2", tmp_path)


def test_memory_scd4(tmp_path: Path) -> None:
    """SCD Type 4 on a partitioned dataset stays within memory bound."""
    _run("scd4", tmp_path)


def test_memory_delta_cdf(tmp_path: Path) -> None:
    """Delta CDF source (file_format='delta') stays within memory bound."""
    _run("delta_cdf", tmp_path)


def test_memory_compact_every(tmp_path: Path) -> None:
    """compact_every triggers compaction without exceeding memory bound."""
    _run("compact_every", tmp_path)


def test_memory_streaming_append_sublinear(tmp_path: Path) -> None:
    """sink_delta streams partition-by-partition; peak RSS is proportional to one
    partition, not the full dataset.  Bound: FIXED_OVERHEAD_MB + MEM_FACTOR *
    (data_mb / N_PARTITIONS).
    """
    _run("streaming_append_sublinear", tmp_path)
