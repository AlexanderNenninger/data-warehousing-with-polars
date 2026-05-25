"""
Munich Monatszahlen demo pipelines.

Ingests three monthly statistical datasets published by the Statistisches Amt
München into Delta Lake tables on S3 using data-warehousing-with-polars.

Three pipelines (all use the same "Monatszahlen Monitoring" CSV schema):

  - accidents   Monatszahlen Verkehrsunfälle — monthly road-accident counts
                (total, with injuries, alcohol-related, hit-and-run).

  - airport     Monatszahlen Flugverkehr — monthly Munich Airport passenger and
                flight-movement statistics.

  - vehicles    Monatszahlen KFZ-Bestand — monthly vehicle fleet by fuel type,
                useful for tracking EV adoption in Munich.

Each source is a single full-history CSV that the portal replaces in-place every
month. The pipelines download it into a date-stamped file so the incremental
watermark sees each monthly run as a new file, then upsert into the Delta table
on the natural keys (MONATSZAHL, AUSPRAEGUNG, JAHR, MONAT).

Data:    Statistisches Amt München
         https://opendata.muenchen.de/pages/monatszahlen-monitoring
License: Datenlizenz Deutschland Namensnennung 2.0
         https://www.govdata.de/dl-de/by-2-0

Required environment variables:
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_REGION               (or AWS_ENDPOINT_URL for non-AWS S3)
    MUNICH_CYCLING_BUCKET    e.g. "my-bucket"
"""

from __future__ import annotations

import logging
import os
import urllib.request
from datetime import datetime
from pathlib import Path

import polars as pl

from data_warehousing_with_polars import incremental, schema

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

BUCKET = os.environ.get("MUNICH_CYCLING_BUCKET", "data-warehousing-with-polars")

# Dataset URLs from the Munich Open Data CKAN portal.
_URLS = {
    "accidents": (
        "https://opendata.muenchen.de/dataset/5e73a82b-7cfb-40cc-9b30-45fe5a3fa24e"
        "/resource/40094bd6-f82d-4979-949b-26c8dc00b9a7/download/verkehrsunfaelle.csv"
    ),
    "airport": (
        "https://opendata.muenchen.de/dataset/9a648dad-0b55-42c7-8ba6-24b7c6bcc599"
        "/resource/ad408efa-528e-409b-bfe2-e1f547992cde/download/flugverkehr.csv"
    ),
    "vehicles": (
        "https://opendata.muenchen.de/dataset/0171c878-0054-495d-b4f8-1947f46dc74a"
        "/resource/b21b2744-b54e-4f11-825e-619431fee648/download/kraftfahrzeuge.csv"
    ),
}

# Stable local staging roots.
_STAGING = Path("/tmp/munich_monatszahlen")

# ── Shared schema ─────────────────────────────────────────────────────────────

# All three "Monatszahlen Monitoring" datasets share this exact schema.
SCHEMA_MONATSZAHLEN = {
    "MONATSZAHL": pl.Utf8,
    "AUSPRAEGUNG": pl.Utf8,
    "JAHR": pl.Int64,
    "MONAT": pl.Utf8,      # "YYYYMM" for monthly rows, "Summe" for annual totals
    "WERT": pl.Float64,    # current value (may be null for not-yet-published months)
    "VORJAHRESWERT": pl.Float64,
    "VERAEND_VORMONAT_PROZENT": pl.Float64,
    "VERAEND_VORJAHRESMONAT_PROZENT": pl.Float64,
    "ZWOELF_MONATE_MITTELWERT": pl.Float64,
}

# Natural merge keys — same for all three datasets.
_MERGE_ON = ["MONATSZAHL", "AUSPRAEGUNG", "JAHR", "MONAT"]


# ── Transform ─────────────────────────────────────────────────────────────────


def _transform(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Parse MONAT into a proper date and add a ``date`` column.

    Monthly rows have MONAT in ``YYYYMM`` format (e.g. ``202601``); annual
    summary rows have MONAT == ``"Summe"``.  Only monthly rows get a real date
    (the 1st of that month); summary rows are excluded so only regular monthly
    observations end up in the Delta table.
    """
    return (
        lf
        .filter(pl.col("MONAT") != "Summe")
        .with_columns(
            pl.col("MONAT")
            .str.strptime(pl.Date, "%Y%m", strict=False)
            .alias("date")
        )
    )


# ── Pipelines ─────────────────────────────────────────────────────────────────


@incremental(
    source=str(_STAGING / "accidents"),
    target=f"s3://{BUCKET}/delta/munich_accidents",
    merge_on=_MERGE_ON,
    file_format="csv",
    reader_kwargs={"null_values": ["NA"], "quote_char": '"'},
)
@schema(expect=SCHEMA_MONATSZAHLEN, on_extra="drop", evolution="cast")
def accidents(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Monthly Munich road-accident counts (total / injuries / alcohol / hit-and-run)."""
    return _transform(lf)


@incremental(
    source=str(_STAGING / "airport"),
    target=f"s3://{BUCKET}/delta/munich_airport",
    merge_on=_MERGE_ON,
    file_format="csv",
    reader_kwargs={"null_values": ["NA"], "quote_char": '"'},
)
@schema(expect=SCHEMA_MONATSZAHLEN, on_extra="drop", evolution="cast")
def airport(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Monthly Munich Airport passenger and flight-movement counts."""
    return _transform(lf)


@incremental(
    source=str(_STAGING / "vehicles"),
    target=f"s3://{BUCKET}/delta/munich_vehicles",
    merge_on=_MERGE_ON,
    file_format="csv",
    reader_kwargs={"null_values": ["NA"], "quote_char": '"'},
)
@schema(expect=SCHEMA_MONATSZAHLEN, on_extra="drop", evolution="cast")
def vehicles(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Monthly Munich vehicle fleet by fuel type (Bestand KFZ-Kraftstoffarten)."""
    return _transform(lf)


# ── Fetch helper ──────────────────────────────────────────────────────────────


def _download(name: str) -> Path:
    """Download the named Monatszahlen CSV into a date-stamped staging file.

    Using a date-stamped filename means each monthly run produces a *new* file,
    so the incremental watermark sees it as fresh input and (re-)upserts the
    latest data into the Delta table.
    """
    dest_dir = _STAGING / name
    dest_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m")
    dest = dest_dir / f"{name}_{stamp}.csv"
    if dest.exists():
        logger.info("Already downloaded today: %s", dest.name)
        return dest

    url = _URLS[name]
    logger.info("Downloading %s …", url)
    req = urllib.request.Request(url, headers={"User-Agent": "data-warehousing-with-polars-demo"})
    with urllib.request.urlopen(req, timeout=60) as resp, dest.open("wb") as fh:
        fh.write(resp.read())
    logger.info("Saved → %s", dest)
    return dest


def fetch_all() -> None:
    """Download the latest CSV for each dataset into the staging directories."""
    for name in _URLS:
        _download(name)


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    fetch_all()

    n = accidents.run()
    logger.info("Accidents pipeline: %d new file(s) ingested.", len(n))

    n = airport.run()
    logger.info("Airport pipeline: %d new file(s) ingested.", len(n))

    n = vehicles.run()
    logger.info("Vehicles pipeline: %d new file(s) ingested.", len(n))


if __name__ == "__main__":
    main()
