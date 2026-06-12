"""Data Warehousing with Polars - A library for data warehousing operations."""

__version__ = "0.1.0"

from data_warehousing_with_polars.incremental import (
    Batch,
    Source,
    from_frame,
    from_query,
    incremental,
)
from data_warehousing_with_polars.schema import schema

__all__ = [
    "Batch",
    "Source",
    "from_frame",
    "from_query",
    "incremental",
    "schema",
]
