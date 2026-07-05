"""Standalone worker script for memory-boundedness checks.

Invoked as::

    python _memory_analysis_impl.py <test_name> <tmp_path>

Each ``run_*`` function is a complete, self-contained memory check that runs
inside a fresh interpreter subprocess.  This avoids the macOS fork-safety
issue where Polars' rayon thread pool, once initialised in the parent pytest
process, deadlocks inside any ``os.fork()``-based child.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make conftest and the installed package importable when run standalone.
sys.path.insert(0, str(Path(__file__).parent))

import polars as pl  # noqa: E402
from _memory_tools import _RSSMeasurement  # noqa
from data_warehousing_with_polars.incremental import incremental  # noqa: E402

# ── Helpers ───────────────────────────────────────────────────────────────────


def _check(delta: float) -> None:
    json.dump(
        {"peak_rss_mb": delta},
        sys.stdout,
    )


# ── Workers ───────────────────────────────────────────────────────────────────


def run_pipeline(tmp_path: str, n_partitions: int, rows_per_partition: int) -> None:
    p = Path(tmp_path)
    src = p / "src"
    src.mkdir()

    @incremental(
        source=str(src),
        target=str(p / "target"),
        # merge_on="measurement",
        merge_on=None,
        partition_by="measurement",
    )
    def pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    m = _RSSMeasurement()
    m.reset()
    pipeline.run()
    delta = m.delta_mb()
    m.stop()
    _check(delta)


def run_pure_polars(tmp_path: str, rows_per_partition: int) -> None:
    p = Path(tmp_path)
    src = p / "src"

    m = _RSSMeasurement()
    m.reset()

    lf = pl.scan_parquet(str(src / "**" / "*.parquet"))
    lf.sink_parquet(
        pl.PartitionBy(
            str(p / "target"),
            key="measurement",
        )
    )

    delta = m.delta_mb()
    m.stop()
    _check(delta)


def run_by_partition(tmp_path: str, rows_per_partition: int) -> None:
    p = Path(tmp_path)
    src = p / "src"

    @incremental(
        source=str(src),
        target=str(p / "target"),
        merge_on=None,
        partition_by="measurement",
        by_partition=True,
        by_partition_workers=1,
    )
    def pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    m = _RSSMeasurement()
    m.reset()

    pipeline.run()

    delta = m.delta_mb()
    m.stop()
    _check(delta)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _name = sys.argv[1]
    _tmp = sys.argv[2]
    _rows_per_partition = int(sys.argv[3])
    globals()[f"run_{_name}"](_tmp, _rows_per_partition)
