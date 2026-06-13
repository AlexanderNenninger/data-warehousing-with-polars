# Data Warehousing with Polars

A [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/) monorepo of [Polars](https://pola.rs)-centric packages.

## Packages

| Package | Path | What it is |
| --- | --- | --- |
| [`data-warehousing-with-polars`](packages/data-warehousing-with-polars/) | `packages/data-warehousing-with-polars/` | Incremental data pipelines on Polars + [Delta Lake](https://delta.io): file tracking, deduplication, SCD semantics, schema validation, and table maintenance. Documented in detail below. |
| [`polars-list-ext`](packages/polars-list-ext/) | `packages/polars-list-ext/` | A Rust Polars expression plugin for List-column signal processing and feature extraction (FFT, Butterworth filters, element-wise aggregation). See [its README](packages/polars-list-ext/README.md). |

The two packages are independent — neither depends on the other.

## Installation

```bash
git clone <repo-url>
cd data-warehousing-with-polars
uv sync --all-extras          # installs both packages into one environment
```

`uv sync` builds the `polars-list-ext` Rust extension, so a [Rust toolchain](https://rustup.rs) is required for a full install. To install only the pure-Python data-warehousing package (no Rust needed):

```bash
uv sync --package data-warehousing-with-polars
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

#### How upserts scale

Upserts are applied by **rewriting only the data a batch touches**, never the whole table, and without using Delta `MERGE` (which mis-handles large match sets):

- **Partitioned target** (`partition_by=` set) — only the partitions present in the incoming batch are rewritten (`replaceWhere`). A batch that lands in a few partitions costs the same regardless of how large the table is, so partitioned tables scale to arbitrary size. Partition by a column your batches are naturally bounded on (e.g. a date or region).
- **Unpartitioned target** — the upsert reads the table, swaps the affected keys, and rewrites it in full. Correct at any size but **O(table)** per run; partition the target if write cost matters.

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

### Fan-in — multiple sources

Pass a list to `source` to read from multiple sources in a single run. Each source produces one `LazyFrame` argument; the transform receives them as positional arguments in the same order as `source`.

```python
@incremental(
    source=["/data/region_a/", "/data/region_b/"],
    target="/data/delta/combined",
    merge_on="id",
)
def combine(lf_a: pl.LazyFrame, lf_b: pl.LazyFrame) -> pl.LazyFrame:
    return pl.concat([lf_a, lf_b])
```

Sources with nothing new are omitted from the call, so the number of arguments can vary between runs. Use `*lfs` when the number of sources is not fixed:

```python
@incremental(source=["/data/a/", "/data/b/", "/data/c/"], target="/data/delta/all", merge_on="id")
def ingest(*lfs: pl.LazyFrame) -> pl.LazyFrame:
    return pl.concat(list(lfs))
```

Fan-in works for **every source type**, each tracked by its own cursor: a list of Delta tables (with `file_format="delta"`), a list of custom `Source` objects, or a **mix** of directories and custom sources in one pipeline:

```python
@incremental(
    source=["/data/uploads/", from_query(fetch_orders, cursor_on="updated_at")],
    target="/data/delta/combined",
    merge_on="id",
)
def combine(*lfs: pl.LazyFrame) -> pl.LazyFrame:
    return pl.concat(list(lfs))
```

The one combination a single list can't express is directories *and* Delta tables together, because `file_format` applies to the whole pipeline — wrap one of them as a custom `Source` if you need that mix.

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

### Custom sources — HTTP, APIs, and other transports

File paths and Delta tables cover the common case, but some sources publish data over HTTP or behind an API. The `Source` protocol lets any object drive a pipeline as long as it implements `poll(since) -> Batch | None`.

**`from_query`** — for sources that return a full snapshot on every call (e.g. a CSV downloaded from a URL). The cursor is the maximum value of a column you designate; `poll` returns `None` when the cursor hasn't advanced.

```python
import urllib.request
import polars as pl
from data_warehousing_with_polars import incremental, from_query


def _fetch(since: object | None) -> pl.LazyFrame | None:
    stamp = "202506"          # e.g. derived from current month
    if since == stamp:
        return None           # already up to date
    with urllib.request.urlopen("https://example.com/data.csv") as r:
        raw = pl.read_csv(r.read())
    return raw.with_columns(pl.lit(stamp).alias("_stamp")).lazy()


@incremental(
    source=from_query(_fetch, cursor_on="_stamp"),
    target="s3://my-bucket/delta/dataset",
    merge_on="id",
)
def dataset(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.drop("_stamp")


dataset.run()
```

The cursor is JSON-serialised and stored in the watermark table. On the next run, `since` receives the value that was returned as `cursor`, so the fetch function can skip unchanged data.

**`from_frame`** — for sources where the transform always re-reads the full frame (e.g. a slowly-changing reference file). The cursor never advances; `merge_on` prevents duplicates.

```python
from data_warehousing_with_polars import incremental, from_frame

@incremental(
    source=from_frame(lambda: pl.scan_csv("https://example.com/reference.csv")),
    target="s3://my-bucket/delta/reference",
    merge_on="id",
)
def reference(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf
```

**Direct `Source` implementation** — implement the protocol directly when the cursor needs custom logic, such as tracking a set of already-processed URLs:

```python
from data_warehousing_with_polars import incremental
from data_warehousing_with_polars.incremental import Batch
import polars as pl


class MyApiSource:
    def poll(self, since: object | None) -> Batch | None:
        seen: set[str] = set(since) if isinstance(since, list) else set()
        new_urls = [u for u in _list_api_urls() if u not in seen]
        if not new_urls:
            return None
        lf = pl.concat([pl.scan_csv(u) for u in new_urls])
        return Batch(frame=lf, cursor=sorted(seen | set(new_urls)))


@incremental(source=MyApiSource(), target="s3://my-bucket/delta/data", merge_on="id")
def data(lf: pl.LazyFrame) -> pl.LazyFrame:
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

Pass `partition_by=` to partition the target table. Partitioning is fixed at table creation, improves query performance when filtering by the partition column, and — for `merge_on` upserts — bounds how much data each run rewrites (see [How upserts scale](#how-upserts-scale)). Partitioning is the lever for upserting into arbitrarily large tables.

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

## polars-list-ext

The `polars-list-ext` package (import `polars_list_ext`) is a Rust [Polars plugin](https://docs.pola.rs/user-guide/plugins/) providing expressions for List-type columns — FFT, windowing, Butterworth filters, element-wise aggregation, interpolation, and range features. It is built and installed by `uv sync`.

```python
import polars as pl
import polars_list_ext as ple

df = pl.DataFrame({"signal": [[0.0, 1.0, 0.0, -1.0] * 4]})
df.with_columns(ple.apply_fft("signal", sample_rate=16).alias("fft"))
```

See the [package README](packages/polars-list-ext/README.md) for the full function list. It is derived from [`polars_list_utils`](https://github.com/dashdeckers/polars_list_utils) by Travis Hammond, modified for use here (attribution in its README).

## Development

```bash
# Run full QA across all packages (format, lint, typecheck, tests)
poe qa

# Individual steps
poe fmt          # ruff format (both packages)
poe lint         # ruff check --fix (both packages)
poe typecheck    # ty check (both packages)
poe test         # data-warehousing pytest (excludes memory tests)
poe test_memory  # memory-boundedness tests (each in a fresh subprocess)
poe test_ext     # polars-list-ext smoke tests
poe ext_build    # rebuild the Rust plugin (maturin develop --release)

# Demo pipelines (require AWS credentials in .env)
poe monatszahlen   # munich_monatszahlen.py
poe cycling        # munich_cycling.py
poe pipelines      # both in sequence

# Docs
poe docs           # great-docs build
poe docs_preview   # great-docs preview (local server)
```

## Requirements

- Python >= 3.12
- polars >= 1.30.0
- deltalake >= 0.22.3
- pyarrow >= 19.0.0
- A [Rust toolchain](https://rustup.rs) to build `polars-list-ext` (not needed if you install only the data-warehousing package)

## License

MIT
