# polars-list-ext

A Polars plugin providing utilities for working with `List`-type columns,
with a focus on signal processing, feature extraction, and general-purpose
functional combinators.

> **Attribution.** This package is derived from
> [`polars_list_utils`](https://github.com/dashdeckers/polars_list_utils) by
> Travis Hammond (dashdeckers), used under the terms of its original license and
> modified for use in this repository. The import name is `polars_list_ext`.

By implementing these operations as a Polars plugin, they participate in query
optimisation and parallelisation rather than falling back to Python-level loops
or leaving the DataFrame.

## Installation

```bash
pip install polars-list-ext
```

## Usage

Free functions are called directly; combinators are accessed through the
`list_ext` expression namespace registered on import:

```python
import polars as pl
import polars_list_ext as ple

# Free function
df.with_columns(ple.apply_fft("signal", sample_rate=1000).alias("spectrum"))

# Namespace combinator
df.with_columns(pl.col("a").list_ext.zip(pl.col("b")).alias("pairs"))
df.with_columns(pl.col("pairs").list_ext.unzip().alias("u")).unnest("u")
```

---

## API Reference

### Signal Processing

| Function | Description |
|---|---|
| `apply_fft(col, sample_rate, ...)` | FFT with optional windowing, Butterworth filter, and normalisation |
| `fft_freqs(n, sample_rate)` | Frequency axis for FFT bins (use with `pl.lit`) |
| `fft_freqs_linspace(start, stop, n)` | Linearly spaced frequency vector (use with `pl.lit`) |

### Feature Extraction

| Function | Description |
|---|---|
| `agg_of_range(y, x, agg, x_min, x_max, ...)` | Aggregate y-values within an x-range |
| `mean_of_range(y, x, x_min, x_max, ...)` | Mean of y-values within an x-range |
| `aggregate_list_col_elementwise(col, list_size, agg)` | Column-wise elementwise aggregation in a GroupBy |

### Arithmetic

| Function | Description |
|---|---|
| `operate_scalar_on_list(list_col, scalar_col, op)` | Apply `add`/`sub`/`mul`/`div` of a scalar column to each list element |
| `interpolate_columns(x_data, y_data, x_interp)` | Interpolate a new y-series at arbitrary x positions |

---

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
