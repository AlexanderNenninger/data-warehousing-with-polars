"""Standalone worker script for memory-boundedness checks.

Invoked as::

    python _memory_impl.py <test_name> <tmp_path>

Each ``run_*`` function is a complete, self-contained memory check that runs
inside a fresh interpreter subprocess.  This avoids the macOS fork-safety
issue where Polars' rayon thread pool, once initialised in the parent pytest
process, deadlocks inside any ``os.fork()``-based child.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make conftest and the installed package importable when run standalone.
sys.path.insert(0, str(Path(__file__).parent))

import polars as pl  # noqa: E402
from conftest import _RSSMeasurement  # noqa  # ty:ignore[unresolved-import]

from data_warehousing_with_polars.incremental import incremental  # noqa: E402

# ── Constants (kept in sync with test_memory.py) ──────────────────────────────

N_ROWS = 400_000
N_PARTITIONS = 5
ROWS_PER_PARTITION = N_ROWS // N_PARTITIONS
FIXED_OVERHEAD_MB = 200.0
MEM_FACTOR = 6.0

# ── Helpers ───────────────────────────────────────────────────────────────────


def _write_partitioned_parquet(src: Path, *, id_offset: int = 0) -> float:
    total_bytes = 0
    for p in range(N_PARTITIONS):
        partition_dir = src / f"date=2024-01-{p + 1:02d}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        start = id_offset + p * ROWS_PER_PARTITION
        ids = list(range(start, start + ROWS_PER_PARTITION))
        df = pl.DataFrame(
            {
                "id": ids,
                "value": [float(i) for i in range(ROWS_PER_PARTITION)],
                "metric_a": [float(i) * 1.1 for i in range(ROWS_PER_PARTITION)],
                "metric_b": [float(i) * 2.2 for i in range(ROWS_PER_PARTITION)],
                "metric_c": [float(i) * 3.3 for i in range(ROWS_PER_PARTITION)],
                "metric_d": [float(i) * 4.4 for i in range(ROWS_PER_PARTITION)],
                "metric_e": [float(i) * 5.5 for i in range(ROWS_PER_PARTITION)],
                "category": [f"category_{i % 10:02d}" for i in ids],
                "name": [f"item_{i:07d}" for i in ids],
                "description": [f"Description for item {i:07d} in partition {p:02d}" for i in ids],
            }
        )
        df.write_parquet(partition_dir / "batch.parquet")
        total_bytes += df.estimated_size()
    return total_bytes / (1024 * 1024)


def _write_partitioned_csv(src: Path) -> float:
    total_bytes = 0
    for p in range(N_PARTITIONS):
        partition_dir = src / f"date=2024-01-{p + 1:02d}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        start = p * ROWS_PER_PARTITION
        df = pl.DataFrame(
            {
                "id": list(range(start, start + ROWS_PER_PARTITION)),
                "value": [float(i) for i in range(ROWS_PER_PARTITION)],
                "metric_a": [float(i) * 1.1 for i in range(ROWS_PER_PARTITION)],
                "metric_b": [float(i) * 2.2 for i in range(ROWS_PER_PARTITION)],
                "metric_c": [float(i) * 3.3 for i in range(ROWS_PER_PARTITION)],
                "metric_d": [float(i) * 4.4 for i in range(ROWS_PER_PARTITION)],
                "metric_e": [float(i) * 5.5 for i in range(ROWS_PER_PARTITION)],
            }
        )
        df.write_csv(partition_dir / "batch.csv")
        total_bytes += df.estimated_size()
    return total_bytes / (1024 * 1024)


def _write_partitioned_ndjson(src: Path) -> float:
    total_bytes = 0
    for p in range(N_PARTITIONS):
        partition_dir = src / f"date=2024-01-{p + 1:02d}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        start = p * ROWS_PER_PARTITION
        df = pl.DataFrame(
            {
                "id": list(range(start, start + ROWS_PER_PARTITION)),
                "value": [float(i) for i in range(ROWS_PER_PARTITION)],
                "metric_a": [float(i) * 1.1 for i in range(ROWS_PER_PARTITION)],
                "metric_b": [float(i) * 2.2 for i in range(ROWS_PER_PARTITION)],
                "metric_c": [float(i) * 3.3 for i in range(ROWS_PER_PARTITION)],
                "metric_d": [float(i) * 4.4 for i in range(ROWS_PER_PARTITION)],
                "metric_e": [float(i) * 5.5 for i in range(ROWS_PER_PARTITION)],
            }
        )
        df.write_ndjson(partition_dir / "batch.ndjson")
        total_bytes += df.estimated_size()
    return total_bytes / (1024 * 1024)


def _check(delta: float, bound: float) -> None:
    print(f"peak_rss={delta:.1f} MB  bound={bound:.1f} MB  ratio={delta / bound:.2f}")
    assert delta < bound, f"peak RSS {delta:.1f} MB >= bound {bound:.1f} MB"


# ── Workers ───────────────────────────────────────────────────────────────────


def run_upsert(tmp_path: str) -> None:
    p = Path(tmp_path)
    src = p / "src"
    src.mkdir()
    data_mb = _write_partitioned_parquet(src)

    @incremental(source=str(src), target=str(p / "target"), merge_on="id")
    def pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.filter(pl.col("value") >= 0)

    m = _RSSMeasurement()
    m.reset()
    pipeline.run()
    delta = m.delta_mb()
    m.stop()
    _check(delta, FIXED_OVERHEAD_MB + MEM_FACTOR * data_mb)


def run_append_only(tmp_path: str) -> None:
    p = Path(tmp_path)
    src = p / "src"
    src.mkdir()
    data_mb = _write_partitioned_parquet(src)

    @incremental(source=str(src), target=str(p / "target"))
    def pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    m = _RSSMeasurement()
    m.reset()
    pipeline.run()
    delta = m.delta_mb()
    m.stop()
    _check(delta, FIXED_OVERHEAD_MB + MEM_FACTOR * data_mb)


def run_fan_in(tmp_path: str) -> None:
    p = Path(tmp_path)
    src_a = p / "src_a"
    src_b = p / "src_b"
    src_a.mkdir()
    src_b.mkdir()
    data_mb = _write_partitioned_parquet(src_a, id_offset=0)
    data_mb += _write_partitioned_parquet(src_b, id_offset=N_ROWS)

    @incremental(
        source=[str(src_a), str(src_b)],
        target=str(p / "target"),
        merge_on="id",
    )
    def pipeline(lf_a: pl.LazyFrame, lf_b: pl.LazyFrame) -> pl.LazyFrame:
        return pl.concat([lf_a, lf_b])

    m = _RSSMeasurement()
    m.reset()
    pipeline.run()
    delta = m.delta_mb()
    m.stop()
    _check(delta, FIXED_OVERHEAD_MB + MEM_FACTOR * data_mb)


def run_csv_format(tmp_path: str) -> None:
    p = Path(tmp_path)
    src = p / "src"
    src.mkdir()
    data_mb = _write_partitioned_csv(src)

    @incremental(
        source=str(src),
        target=str(p / "target"),
        merge_on="id",
        file_format="csv",
    )
    def pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    m = _RSSMeasurement()
    m.reset()
    pipeline.run()
    delta = m.delta_mb()
    m.stop()
    _check(delta, FIXED_OVERHEAD_MB + MEM_FACTOR * data_mb)


def run_ndjson_format(tmp_path: str) -> None:
    p = Path(tmp_path)
    src = p / "src"
    src.mkdir()
    data_mb = _write_partitioned_ndjson(src)

    @incremental(
        source=str(src),
        target=str(p / "target"),
        merge_on="id",
        file_format="ndjson",
    )
    def pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    m = _RSSMeasurement()
    m.reset()
    pipeline.run()
    delta = m.delta_mb()
    m.stop()
    _check(delta, FIXED_OVERHEAD_MB + MEM_FACTOR * data_mb)


def run_scd2(tmp_path: str) -> None:
    p = Path(tmp_path)
    src = p / "src"
    src.mkdir()
    data_mb = _write_partitioned_parquet(src)

    @incremental(
        source=str(src),
        target=str(p / "target"),
        merge_on="id",
        scd_type=2,
    )
    def pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    m = _RSSMeasurement()
    m.reset()
    pipeline.run()
    delta = m.delta_mb()
    m.stop()
    _check(delta, FIXED_OVERHEAD_MB + MEM_FACTOR * data_mb)


def run_scd4(tmp_path: str) -> None:
    p = Path(tmp_path)
    src = p / "src"
    src.mkdir()
    data_mb = _write_partitioned_parquet(src)

    @incremental(
        source=str(src),
        target=str(p / "target"),
        history_target=str(p / "history"),
        merge_on="id",
        scd_type=4,
    )
    def pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    m = _RSSMeasurement()
    m.reset()
    pipeline.run()
    delta = m.delta_mb()
    m.stop()
    _check(delta, FIXED_OVERHEAD_MB + MEM_FACTOR * data_mb)


def run_delta_cdf(tmp_path: str) -> None:
    from deltalake import write_deltalake  # noqa: PLC0415

    p = Path(tmp_path)
    source = str(p / "source")
    data_mb = 0.0
    for part in range(N_PARTITIONS):
        start = part * ROWS_PER_PARTITION
        ids = list(range(start, start + ROWS_PER_PARTITION))
        df = pl.DataFrame(
            {
                "id": ids,
                "value": [float(i) for i in range(ROWS_PER_PARTITION)],
                "metric_a": [float(i) * 1.1 for i in range(ROWS_PER_PARTITION)],
                "metric_b": [float(i) * 2.2 for i in range(ROWS_PER_PARTITION)],
                "metric_c": [float(i) * 3.3 for i in range(ROWS_PER_PARTITION)],
                "metric_d": [float(i) * 4.4 for i in range(ROWS_PER_PARTITION)],
                "metric_e": [float(i) * 5.5 for i in range(ROWS_PER_PARTITION)],
                "category": [f"category_{i % 10:02d}" for i in ids],
                "name": [f"item_{i:07d}" for i in ids],
                "description": [
                    f"Description for item {i:07d} in partition {part:02d}" for i in ids
                ],
            }
        )
        write_deltalake(
            source,
            df.to_arrow(),
            mode="append" if part > 0 else "overwrite",
            configuration={"delta.enableChangeDataFeed": "true"},
        )
        data_mb += df.estimated_size() / (1024 * 1024)

    @incremental(
        source=source,
        target=str(p / "target"),
        merge_on="id",
        file_format="delta",
    )
    def pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    m = _RSSMeasurement()
    m.reset()
    pipeline.run()
    delta = m.delta_mb()
    m.stop()
    _check(delta, FIXED_OVERHEAD_MB + MEM_FACTOR * data_mb)


def run_compact_every(tmp_path: str) -> None:
    p = Path(tmp_path)
    src = p / "src"
    src.mkdir()
    data_mb = _write_partitioned_parquet(src)

    @incremental(
        source=str(src),
        target=str(p / "target"),
        merge_on="id",
        compact_every=1,
    )
    def pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    m = _RSSMeasurement()
    m.reset()
    pipeline.run()
    delta = m.delta_mb()
    m.stop()
    _check(delta, FIXED_OVERHEAD_MB + MEM_FACTOR * data_mb)


def run_streaming_append_sublinear(tmp_path: str) -> None:
    p = Path(tmp_path)
    target = str(p / "target")

    # Seed the target table so the real pipeline performs a streaming append.
    seed_src = p / "seed_src"
    seed_src.mkdir()
    pl.DataFrame(
        {
            "id": [-1],
            "value": [-1.0],
            "metric_a": [-1.0],
            "metric_b": [-1.0],
            "metric_c": [-1.0],
            "metric_d": [-1.0],
            "metric_e": [-1.0],
            "category": ["_seed"],
            "name": ["_seed_0000001"],
            "description": ["_seed_row"],
        }
    ).write_parquet(seed_src / "seed.parquet")

    @incremental(source=str(seed_src), target=target)
    def pipeline_seed(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    pipeline_seed.run()

    src = p / "src"
    src.mkdir()
    data_mb = _write_partitioned_parquet(src)

    @incremental(source=str(src), target=target)
    def pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    m = _RSSMeasurement()
    m.reset()
    pipeline.run()
    delta = m.delta_mb()
    m.stop()
    _check(delta, FIXED_OVERHEAD_MB + MEM_FACTOR * (data_mb / N_PARTITIONS))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _name = sys.argv[1]
    _tmp = sys.argv[2]
    globals()[f"run_{_name}"](_tmp)
