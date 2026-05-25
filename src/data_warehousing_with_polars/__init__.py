"""Data Warehousing with Polars - A library for data warehousing operations."""

__version__ = "0.1.0"

from data_warehousing_with_polars.incremental import incremental
from data_warehousing_with_polars.schema import schema

__all__ = [
    "incremental",
    "schema",
]
