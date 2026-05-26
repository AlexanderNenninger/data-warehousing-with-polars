# Data Warehousing with Polars

A Python library for building incremental data pipelines on top of [Polars](https://pola.rs) and [Delta Lake](https://delta.io). Handles file tracking, deduplication, SCD semantics, schema validation, and table maintenance so transform functions stay focused on business logic.

## Installation

```bash
git clone <repo-url>
cd data-warehousing-with-polars
uv pip install -e .
```

## Concepts

### Incremental pipelines

The `@incremental` decorator wraps a `LazyFrame → LazyFrame` function and turns it into an `IncrementalPipeline`. Each call to `.run()` processes only files that have not been seen before. Already-processed files are skipped automatically.

```python
from data_warehousing_with_polars import incremental
import polars as pl


@incremental(source="/data/uploads/", target="/data/delta/clean", merge_on="id")
def clean(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.filter(pl.col("value") > 0)


clean.run()  # process new files, write to target
clean.run(dry_run=True)  # list new files without reading or writing
clean.status()  # watermark table as a DataFrame
clean.reset()  # clear watermark; next run reprocesses all files
```

Supported file formats: `"parquet"` (default), `"csv"`, `"ndjson"`, `"delta"`.

### Watermark

The watermark is a Delta table that records every file path (or Delta version) that has been successfully ingested. It is written *after* a successful target write, so a crashed run simply reprocesses the same files on the next call to `.run()`.

The watermark is stored at `target/.watermark` by default. Override with `watermark_store=`.

### Merge semantics (SCD Type 1)

When `merge_on` is set, each `.run()` performs an upsert: existing rows with matching keys are updated, new keys are inserted. This is SCD Type 1 — the table always holds the latest state with no history.

```python
@incremental(source="/data/uploads/", target="/data/delta/users", merge_on="id")
def users(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf
```

Set `merge_on=None` for append-only mode (no deduplication):

```python
@incremental(source="/data/logs/", target="/data/delta/events")
def events(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf
```

### SCD Type 2 — full row history

`scd_type=2` keeps the complete history of every row. Each time a key reappears, the old row is closed (`is_current = false`, `valid_to = <now>`) and a new row is appended (`is_current = true`, `valid_to = null`).

The three bookkeeping columns — `valid_from`, `valid_to`, `is_current` — are injected automatically; the transform function does not need to produce them.

```python
@incremental(
    source="/data/uploads/",
    target="/data/delta/users_history",
    merge_on="id",
    scd_type=2,
)
def users(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf
```

Querying current state:

```python
pl.scan_delta("/data/delta/users_history").filter(pl.col("is_current"))
```

Querying point-in-time state:

```python
from datetime import datetime, timezone

cutoff = datetime(2026, 1, 1, tzinfo=timezone.utc)
(
    pl.scan_delta("/data/delta/users_history").filter(
        (pl.col("valid_from") <= cutoff)
        & (pl.col("valid_to").is_null() | (pl.col("valid_to") > cutoff))
    )
)
```

Re-running `.run()` with the same input batch is safe: already-closed rows are skipped and already-appended rows are deduplicated on `(merge_on, valid_from)`.

### SCD Type 4 — separate history table

`scd_type=4` uses two tables: a main table that always holds only the current state (like SCD Type 1), and a separate history table that accumulates all superseded versions with a `superseded_at` timestamp. Requires `history_target=`.

```python
@incremental(
    source="/data/uploads/",
    target="/data/delta/users",
    history_target="/data/delta/users_history",
    merge_on="id",
    scd_type=4,
)
def users(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf
```

The main table stays small and fast to query. The history table is append-only and accumulates one row per superseded version.

### Fan-in — multiple source directories

Pass a list of paths to `source` to read from multiple directories in a single run. Each source directory produces one `LazyFrame` argument; the transform receives them as positional arguments in the same order as `source`.

```python
@incremental(
    source=["/data/region_a/", "/data/region_b/"],
    target="/data/delta/combined",
    merge_on="id",
)
def combine(lf_a: pl.LazyFrame, lf_b: pl.LazyFrame) -> pl.LazyFrame:
    return pl.concat([lf_a, lf_b])
```

Sources with no new files are omitted from the call, so the number of arguments can vary between runs. Use `*lfs` when the number of sources is not fixed:

```python
@incremental(source=["/data/a/", "/data/b/", "/data/c/"], target="/data/delta/all", merge_on="id")
def ingest(*lfs: pl.LazyFrame) -> pl.LazyFrame:
    return pl.concat(list(lfs))
```

### Delta source — Change Data Feed

When `file_format="delta"`, the pipeline reads from another Delta table using the [Change Data Feed](https://docs.delta.io/latest/delta-change-data-feed.html) (CDF). Only rows that changed since the last processed version are read; the first run loads the full table.

CDF metadata columns (`_change_type`, `_commit_version`, `_commit_timestamp`) are dropped before the transform is called.

```python
@incremental(
    source="/data/delta/source",
    target="/data/delta/sink",
    merge_on="id",
    file_format="delta",
)
def propagate(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf
```

### Schema validation

The `@schema` decorator validates a LazyFrame's schema before it reaches the transform. It operates on schema metadata only (via `lf.collect_schema()`), keeping the pipeline lazy on the happy path.

```python
from data_warehousing_with_polars import schema, SchemaError


@incremental(source="/data/uploads/", target="/data/delta/clean", merge_on="id")
@schema(
    expect={"id": pl.Int64, "value": pl.Float64, "ts": pl.Datetime("us", "UTC")},
    on_missing="quarantine",
    on_extra="drop",
    evolution="cast",
    quarantine="/data/delta/bad_batches",
)
def clean(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.filter(pl.col("value") > 0)
```

**`on_missing`** — what to do when an expected column is absent:
- `"raise"` (default) — raise `SchemaError`
- `"drop"` — silently skip the batch, return an empty LazyFrame
- `"quarantine"` — write the batch to a quarantine Delta table and skip it

**`on_extra`** — what to do when unexpected columns are present:
- `"ignore"` (default) — pass them through
- `"raise"` — raise `SchemaError`
- `"drop"` — remove them before the transform

**`evolution`** — what to do when a column has the wrong dtype:
- `"strict"` (default) — raise `SchemaError`
- `"cast"` — attempt a Polars cast to the expected type
- `"merge"` — same as `"cast"`, mirrors Delta's `schema_mode="merge"` semantics

Columns prefixed with `_` (e.g. `_source_file`, `_ingested_at`) are injected by the library and are always exempt from `on_extra` checks.

Catching violations manually:

```python
try:
    clean(raw_lf).collect()
except SchemaError as e:
    for v in e.violations:
        print(v.column, v.issue, v.expected, v.actual)
```

### Partitioning

Pass `partition_by=` to partition the target table. Partitioning is fixed at table creation and improves query performance on large tables when filtering by the partition column.

```python
@incremental(
    source="/data/uploads/",
    target="/data/delta/events",
    merge_on="id",
    partition_by="region",
)
def events(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf
```

### Compaction and vacuuming

Delta Lake accumulates small files over time. The `maintain()` method (and the `compact_every` trigger) consolidate them.

Manual maintenance:

```python
clean.maintain(
    compact=True,  # coalesce small files (OPTIMIZE)
    z_order_by="region",  # Z-order by column for better query pruning
    vacuum=True,  # delete files older than retention_hours
    retention_hours=168,  # default: 7 days
)
```

Automatic compaction every N successful runs:

```python
@incremental(source="/data/uploads/", target="/data/delta/clean", merge_on="id", compact_every=20)
def clean(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf
```

## Development

```bash
# Run full QA (format, lint, typecheck, tests)
poe qa

# Individual steps
poe fmt          # ruff format
poe lint         # ruff check --fix
poe typecheck    # ty check
poe test         # pytest (excludes memory tests)
poe test_memory  # memory-boundedness tests (each in a fresh subprocess)
```

## Requirements

- Python >= 3.12
- polars >= 1.41.0
- deltalake >= 0.22.3
- pyarrow >= 19.0.0

## License

MIT
