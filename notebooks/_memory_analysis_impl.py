"""Standalone worker script for memory-boundedness checks.

Invoked as::

    python _memory_analysis_impl.py <test_name> <tmp_path> <n_partitions> <rows_per_partition>

Each ``run_*`` function is a complete, self-contained memory check that runs
inside a fresh interpreter subprocess.  This avoids the macOS fork-safety
issue where Polars' rayon thread pool, once initialised in the parent pytest
process, deadlocks inside any ``os.fork()``-based child.

Every worker is deliberately simple: define an ``@incremental`` pipeline as a
local function, reset the RSS tracker, call ``pipeline.run()`` exactly once,
and report the result. All source dataset creation — including any prior
runs needed to establish watermark/target state, second batches, fan-in
sources, or pre-existing file fragmentation for compaction — happens in the
notebook (the parent process), not here. Building that data in the same
process as the measured run previously inflated the RSS baseline the measured
run resets against: ``m.reset()`` only zeroes the delta, it can't make the OS
reclaim pages the allocator already grabbed for setup data, so the measured
run's real growth could land entirely inside that already-resident headroom
and read back as ~0 even for hundreds of MB of genuine work.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make conftest and the installed package importable when run standalone.
sys.path.insert(0, str(Path(__file__).parent))

import polars as pl  # noqa: E402
from _memory_tools import _RSSMeasurement  # noqa
from data_warehousing_with_polars.incremental import _DeltaCdfSource, incremental  # noqa: E402

# ── Helpers ───────────────────────────────────────────────────────────────────


def _check(delta: float) -> None:
    json.dump({"peak_rss_mb": delta}, sys.stdout)


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

    Assumes the notebook has already created ``cdf_source``, run this same
    pipeline once to establish the watermark, and appended the new CDF-visible
    commit this run picks up as its single-use batch.
    """
    p = Path(tmp_path)
    source = p / "cdf_source"

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

    m = _RSSMeasurement()
    m.reset()

    pipeline.run()

    delta = m.delta_mb()
    m.stop()
    _check(delta)


def run_upsert(tmp_path: str, n_partitions: int, rows_per_partition: int) -> None:
    """Plain upsert — ``merge_on`` set, default ``scd_type=1``, ``by_partition=True``.

    The most common real usage of ``merge_on``: no SCD history tracking, just
    "keep the latest row per key". Goes through ``_upsert_overwrite``, which
    already folds "existing rows not matching new keys" + "new rows" into a
    single ``sink_delta`` commit — the same pattern retrofitted onto SCD2 — so
    this is here to confirm that empirically rather than infer it from reading
    the code.

    Assumes the notebook has already created the table (a first run against
    the harness's ``src``) and written the new files this run's
    ``_upsert_overwrite`` call actually processes.

    Merges on ``"channel"`` alone, not ``["measurement", "channel"]`` like
    :func:`run_scd4`: ``_sink_target`` already builds
    ``join_cols = keys + partition_list`` internally, so including the
    partition column in ``merge_on`` too would duplicate it. Unlike SCD4,
    ``_upsert_overwrite`` never dedupes the incoming batch (it only anti-joins
    existing rows against it), so there's no collapse-to-trivial-size risk from
    ``channel``'s low cardinality here.
    """
    p = Path(tmp_path)

    @incremental(
        source=str(p / "src"),
        target=str(p / "target"),
        merge_on="channel",
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


def run_fan_in_cdf_and_file(tmp_path: str, n_partitions: int, rows_per_partition: int) -> None:
    """``by_partition=True`` fanning in two active sources of different kinds at once:
    a re-scannable file glob and a single-use Delta CDF ``RecordBatchReader``.

    ``_run_by_partition`` resolves each source slot independently (a plain lazy
    frame for the file source, a per-combo ``load_cdf(predicate=...)`` factory for
    the CDF source — see ``run_by_partition_cdf``); every other worker here only
    ever has one active source, so this is the only case exercising both code
    paths together in the same batch, with combo enumeration merged across them.

    Assumes the notebook has already created both sources, run this pipeline
    once to establish both cursors, and written new data to both.
    """
    p = Path(tmp_path)

    @incremental(
        source=[str(p / "src"), _DeltaCdfSource(str(p / "cdf_source"))],
        target=str(p / "target"),
        merge_on=None,
        partition_by="measurement",
        by_partition=True,
        by_partition_workers=1,
    )
    def pipeline(lf_file: pl.LazyFrame, lf_cdf: pl.LazyFrame) -> pl.LazyFrame:
        return pl.concat([lf_file, lf_cdf], how="diagonal_relaxed")

    m = _RSSMeasurement()
    m.reset()

    pipeline.run()

    delta = m.delta_mb()
    m.stop()
    _check(delta)


def run_compact_every(tmp_path: str, n_partitions: int, rows_per_partition: int) -> None:
    """``by_partition=True`` with ``compact_every=1``, triggering ``maintain()``
    (OPTIMIZE + VACUUM) right after the run — a code path (``maintenance.py``)
    none of the other workers touch at all.

    Assumes the notebook has already accumulated several small, uncompacted
    prior commits into ``target`` (simulating real file fragmentation from
    repeated small batches), so this run's ``maintain()`` call has real work
    to do rather than optimising a freshly-created, already-tidy table.
    """
    p = Path(tmp_path)

    @incremental(
        source=str(p / "src"),
        target=str(p / "target"),
        merge_on=None,
        partition_by="measurement",
        by_partition=True,
        by_partition_workers=1,
        compact_every=1,
    )
    def pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    m = _RSSMeasurement()
    m.reset()

    # compact_every=1 triggers maintain(target) internally right after this run.
    pipeline.run()

    delta = m.delta_mb()
    m.stop()
    _check(delta)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _name = sys.argv[1]
    _tmp = sys.argv[2]
    _n_partitions = int(sys.argv[3])
    _rows_per_partition = int(sys.argv[4])
    globals()[f"run_{_name}"](_tmp, _n_partitions, _rows_per_partition)
