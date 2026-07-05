"""Standalone worker script for memory-boundedness checks.

Invoked as::

    python _memory_analysis_impl.py <test_name> <tmp_path> <n_partitions> <rows_per_partition>

Each ``run_*`` function is a complete, self-contained memory check that runs
inside a fresh interpreter subprocess.  This avoids the macOS fork-safety
issue where Polars' rayon thread pool, once initialised in the parent pytest
process, deadlocks inside any ``os.fork()``-based child.

Every worker takes the same ``(tmp_path, n_partitions, rows_per_partition)``
signature so they can all be run with identical parameters and compared on
one chart, even workers (like ``run_by_partition_cdf``) that build their own
data instead of using the harness-written ``src`` directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make conftest and the installed package importable when run standalone.
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from _memory_tools import _RSSMeasurement  # noqa
from data_warehousing_with_polars.incremental import incremental  # noqa: E402
from deltalake import write_deltalake  # noqa: E402

# ── Helpers ───────────────────────────────────────────────────────────────────


def _check(delta: float, dataset_size_mb: float | None = None) -> None:
    """Report peak RSS delta, and optionally this worker's own dataset size.

    Most workers process the harness-written ``src`` directory, whose size the
    notebook already knows (from ``write_partitioned_measurements``). Workers
    that build their own data instead (e.g. ``run_by_partition_cdf``, which
    ignores ``src`` entirely) report their real size here so the notebook
    doesn't silently mislabel it with the size of unrelated, unused data.
    """
    payload: dict[str, float] = {"peak_rss_mb": delta}
    if dataset_size_mb is not None:
        payload["dataset_size_mb"] = dataset_size_mb
    json.dump(payload, sys.stdout)


# ── Workers ───────────────────────────────────────────────────────────────────


def run_pure_polars(tmp_path: str, n_partitions: int, rows_per_partition: int) -> None:
    p = Path(tmp_path)
    src = p / "src"

    m = _RSSMeasurement()
    m.reset()

    lf = pl.scan_parquet(str(src / "**" / "*.parquet"))
    lf.sink_delta(
        str(p / "target"),
        delta_write_options={"partition_by": ["measurement"]},
    )

    delta = m.delta_mb()
    m.stop()
    _check(delta)


def run_all(tmp_path: str, n_partitions: int, rows_per_partition: int) -> None:
    p = Path(tmp_path)
    src = p / "src"

    @incremental(
        source=str(src),
        target=str(p / "target"),
        merge_on=None,
        partition_by="measurement",
        by_partition=False,
    )
    def pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    m = _RSSMeasurement()
    m.reset()

    pipeline.run()

    delta = m.delta_mb()
    m.stop()
    _check(delta)


def run_by_partition(tmp_path: str, n_partitions: int, rows_per_partition: int) -> None:
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


def run_scd2(tmp_path: str, n_partitions: int, rows_per_partition: int) -> None:
    """First-write SCD2, ``by_partition=True``.

    SCD2 keeps every row (full history), so its output size is inherently
    proportional to input size — unlike SCD4, there's no dedup to shrink it.
    With ``by_partition=False`` that means one unpartitioned write handling
    the whole batch in a single pass, same "no backpressure" shape as `all`
    (see the ``_sink_scd2`` table-creation branch, which now streams via
    ``sink_delta`` instead of ``.collect()`` + ``write_deltalake()``, but that
    alone doesn't bound peak memory to less than the whole dataset).
    ``by_partition=True`` bounds it to one partition's worth at a time
    instead, same as the plain-append ``by_partition`` worker.
    """
    p = Path(tmp_path)
    src = p / "src"

    @incremental(
        source=str(src),
        target=str(p / "target"),
        merge_on="channel",
        scd_type=2,
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


def run_scd4(tmp_path: str, n_partitions: int, rows_per_partition: int) -> None:
    """First-write SCD4, unpartitioned execution (``by_partition=False``).

    Same as :func:`run_scd2`, for the ``_sink_scd4`` table-creation branch
    (which also used to eagerly ``.collect()`` + dedup before writing).

    Merges on ``["measurement", "channel"]`` rather than ``"channel"`` alone:
    ``channel`` only has 5 distinct values regardless of scale, so merging on
    it alone would let ``_sink_scd4``'s ``.unique(subset=keys, keep="last")``
    collapse the whole batch down to ~5 rows before ever writing — trivial at
    any dataset size, and not actually exercising the first-write path at
    scale. The composite key's cardinality scales with ``n_partitions``.
    """
    p = Path(tmp_path)
    src = p / "src"

    @incremental(
        source=str(src),
        target=str(p / "target"),
        history_target=str(p / "target_history"),
        merge_on=["measurement", "channel"],
        scd_type=4,
        partition_by="measurement",
        by_partition=False,
    )
    def pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    m = _RSSMeasurement()
    m.reset()

    pipeline.run()

    delta = m.delta_mb()
    m.stop()
    _check(delta)


def run_by_partition_cdf(tmp_path: str, n_partitions: int, rows_per_partition: int) -> None:
    """``by_partition=True`` reading a Delta *Change Data Feed* source.

    A CDF batch is a one-shot ``RecordBatchReader`` stream, so it can't be
    lazily re-scanned once per partition combo the way a file source can.
    ``_run_by_partition`` used to fully ``.collect()`` it once up front (later
    changed to a local Parquet spill instead); it now re-invokes
    ``DeltaTable.load_cdf()`` fresh per partition combo with a SQL predicate
    scoped to that partition, so delta-rs prunes at the source — no
    materialisation and no local disk copy either.

    Ignores the harness-provided ``src`` directory entirely — it builds its
    own CDF source (with the same ``n_partitions``/``rows_per_partition`` as
    every other worker in the sweep) instead, since it needs two versions of
    one Delta table rather than a directory of files. Reports its own dataset
    size (the second, CDF-triggering commit) via ``_check`` rather than the
    harness's unused ``src`` size.
    """
    p = Path(tmp_path)
    source = p / "cdf_source"

    def _seed() -> pl.DataFrame:
        return pl.concat(
            [
                pl.DataFrame(
                    {
                        "measurement": [f"measurement_{i}"] * rows_per_partition,
                        "value": np.random.rand(rows_per_partition),
                    }
                )
                for i in range(n_partitions)
            ]
        )

    write_deltalake(
        str(source),
        _seed().to_arrow(),
        mode="overwrite",
        partition_by=["measurement"],
        configuration={"delta.enableChangeDataFeed": "true"},
    )

    @incremental(
        source=str(source),
        target=str(p / "target"),
        file_format="delta",
        merge_on=None,
        partition_by="measurement",
        by_partition=True,
        by_partition_workers=1,
    )
    def pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    # First run: from_version=None, full scan_delta — establishes the watermark
    # cursor so the second run below takes the CDF/single-use branch.
    pipeline.run()

    # A second CDF-visible commit, same size as the first, so the second run's
    # single-use RecordBatchReader batch is genuinely large across all partitions.
    new_batch = _seed()
    write_deltalake(str(source), new_batch.to_arrow(), mode="append", partition_by=["measurement"])

    m = _RSSMeasurement()
    m.reset()

    pipeline.run()

    delta = m.delta_mb()
    m.stop()
    _check(delta, dataset_size_mb=new_batch.estimated_size(unit="mb"))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _name = sys.argv[1]
    _tmp = sys.argv[2]
    _n_partitions = int(sys.argv[3])
    _rows_per_partition = int(sys.argv[4])
    globals()[f"run_{_name}"](_tmp, _n_partitions, _rows_per_partition)
