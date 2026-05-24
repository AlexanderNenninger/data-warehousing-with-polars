"""
record_batch_source.py

A single function: make_lazy. Wraps a PyArrow RecordBatchReader in a
streaming Polars LazyFrame via the Polars IO plugin API (register_io_source).

This is infrastructure — a low-level adapter between the Arrow and Polars
type systems — with no conceptual relationship to incremental pipelines, SCD,
schema validation, or compaction. It is used only by _read_delta_source when
reading the Delta Change Data Feed, which returns an Arrow RecordBatchReader
via DeltaTable.load_cdf().

The reason it is separated rather than inlined into incremental.py is that
the register_io_source pattern is non-obvious and worth isolating for
readability and testability. It is also the most likely part of the library
to need updating as the Polars IO plugin API evolves.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Generator

import polars as pl
from polars.io.plugins import register_io_source

if TYPE_CHECKING:
    import pyarrow as pa


def make_lazy(reader: "pa.RecordBatchReader") -> pl.LazyFrame:
    """Wrap a PyArrow ``RecordBatchReader`` in a streaming Polars ``LazyFrame``.

    Each call to ``reader.read_next_batch()`` yields one Arrow
    ``RecordBatch``, which is converted to a Polars ``DataFrame`` and yielded
    to the Polars streaming engine one morsel at a time. No batch is held in
    memory beyond what the current engine stage requires.

    Supports predicate pushdown, column projection, and ``n_rows`` limiting
    via the ``source_gen`` callback signature, so the Polars optimiser can
    apply filters and projections before data is yielded rather than after.

    Args:
        reader: A PyArrow ``RecordBatchReader``, typically from
                ``DeltaTable.load_cdf()``.

    Returns:
        A ``LazyFrame`` backed by the streaming reader. The schema is
        derived from ``reader.schema`` so that downstream operations can
        be planned without reading any data.
    """

    _schema_frame = pl.from_arrow(reader.schema.empty_table())
    assert isinstance(_schema_frame, pl.DataFrame)
    polars_schema = _schema_frame.schema

    def _source_gen(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Generator[pl.DataFrame, None, None]:
        rows_yielded = 0
        while True:
            try:
                batch = reader.read_next_batch()
            except StopIteration:
                return
            _batch_df = pl.from_arrow(batch)
            assert isinstance(_batch_df, pl.DataFrame)
            df = _batch_df
            if with_columns is not None:
                present = [c for c in with_columns if c in df.columns]
                df = df.select(present)
            if n_rows is not None:
                remaining = n_rows - rows_yielded
                if remaining <= 0:
                    return
                df = df.head(remaining)
            if len(df) == 0:
                continue
            rows_yielded += len(df)
            yield df
            if n_rows is not None and rows_yielded >= n_rows:
                return

    return register_io_source(_source_gen, schema=polars_schema)
