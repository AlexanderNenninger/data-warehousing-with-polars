from pathlib import Path
from urllib.parse import quote

import numpy as np
import polars as pl


def write_partitioned_measurements(
    src: Path, n_partitions: int, rows_per_partition: int, suffix: str = "00001"
) -> float:
    """Write partitioned parquet with (measurement, channel, value) columns.

    ``suffix`` names the file within each partition dir, so calling this again
    with a different suffix adds a genuinely new file for a later run to pick
    up (rather than overwriting the first batch — same path, same watermark
    entry, so it would never be seen as "new").
    """
    total_mbytes = 0.0
    for p in range(n_partitions):
        measurement = f"measurement_{p}"
        partition_dir = src / f"measurement={quote(measurement)}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        df = pl.DataFrame([
            pl.repeat(measurement, rows_per_partition, eager=True).rename("measurement"),
            (pl.int_range(0, rows_per_partition, eager=True) % 5)
            .cast(pl.String)
            .str.pad_start(length=3, fill_char="0")
            .rename("channel"),
            pl.Series(np.random.rand(rows_per_partition)).rename("value"),
        ])
        df.write_parquet(partition_dir / f"{suffix}.parquet")
        total_mbytes += df.estimated_size(unit="mb")
    return total_mbytes
