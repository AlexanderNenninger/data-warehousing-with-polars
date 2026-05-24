"""Data Warehousing with Polars - A library for data warehousing operations."""

__version__ = "0.1.0"

from data_warehousing_with_polars.incremental import IncrementalPipeline, incremental
from data_warehousing_with_polars.schema import SchemaError, SchemaViolation, schema

__all__ = [
    "incremental",
    "IncrementalPipeline",
    "schema",
    "SchemaError",
    "SchemaViolation",
]
