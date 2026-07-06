# polars-list-ext

A Polars plugin providing general-purpose functional combinators for working
with `List`-type columns.

By implementing these operations as a Polars plugin, they participate in query
optimisation and parallelisation rather than falling back to Python-level loops
or leaving the DataFrame.

## Installation

```bash
pip install polars-list-ext
```

## Usage

Combinators are accessed through the `list_ext` expression namespace,
registered on import:

```python
import polars as pl
import polars_list_ext  # noqa: F401 — registers the namespace

df.with_columns(pl.col("a").list_ext.zip(pl.col("b")).alias("pairs"))
df.with_columns(pl.col("pairs").list_ext.unzip().alias("u")).unnest("u")
```

---

## API Reference

### Combinators — `pl.col(...).list_ext.*`

All combinators are accessed via the `list_ext` expression namespace. Importing
`polars_list_ext` registers it automatically.

#### Structural

| Method | Description |
|---|---|
| `.enumerate()` | Pair each element with its index → `List[Struct{index: UInt32, value: T}]` |
| `.dedup()` | Remove consecutive duplicate elements (like Unix `uniq`) |
| `.rotate(n)` | Circular shift by `n` positions (positive = right, negative = left) |
| `.flat_map(op, value)` | Apply a scalar arithmetic op then return a flat `List[Float64]` |

#### Windowing / Chunking

| Method | Description |
|---|---|
| `.windows(size, step=1)` | Sliding window view → `List[List[T]]` |
| `.chunks(size)` | Non-overlapping partitions → `List[List[T]]`; last chunk may be smaller |

#### Searching

| Method | Description |
|---|---|
| `.position(op, value)` | Index of first element matching `op` (`"eq"/"ne"/"gt"/"ge"/"lt"/"le"`) → `UInt32` or `null` |

#### Zipping / Pairing

| Method | Description |
|---|---|
| `.zip(other)` | Pair elements from two lists → `List[Struct{first: T, second: U}]` |
| `.unzip()` | Split a `List[Struct]` into a `Struct` of lists — mirrors `struct.unnest` |

#### Joining

| Method | Description |
|---|---|
| `.join(other, on, how, suffix="_right")` | Key-based join on `List[Struct]` rows; `how`: `"inner"/"left"/"anti"` |

---

## Development

### Prerequisites

- Rust (via `rustup`)
- Python ≥ 3.12 (via `uv`)

### Setup

```bash
# from the monorepo root
uv sync
uv run poe ext_build   # compile the Rust plugin
```

### Tasks

```bash
uv run poe ext_build   # compile
uv run poe test_ext    # run tests (smoke + property)
uv run poe lint        # ruff
```
