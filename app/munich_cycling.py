"""
Munich Raddauerzählstellen demo pipeline.

Fetches monthly cycling-counter CSVs from the Munich Open Data Portal and
loads them into Delta tables on S3 using data-warehousing-with-polars.

Two pipelines:
  - counts_15min  Append-only 15-minute interval counts per counting station.
                  Watermark prevents double-processing.
  - counts_daily  Daily totals with weather data (temperature, precipitation,
                  cloud cover, sunshine hours). Upserts on (datum, zaehlstelle)
                  so weather values that are revised retroactively are corrected.

Data:    Raddauerzählstellen München (Mobilitätsreferat LHM), 2022–present
         https://opendata.muenchen.de/dataset/daten-der-raddauerzaehlstellen-muenchen-{year}
License: Datenlizenz Deutschland Namensnennung 2.0
         https://www.govdata.de/dl-de/by-2-0

Required environment variables:
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_REGION               (or AWS_ENDPOINT_URL for non-AWS S3)
    MUNICH_CYCLING_BUCKET    e.g. "my-bucket"
"""

from __future__ import annotations

import csv
import io
import json
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
FIRST_YEAR = 2022  # earliest year available on the portal
CURRENT_YEAR = datetime.now().year
CKAN_API = "https://opendata.muenchen.de/api/3/action/package_show?id=daten-der-raddauerzaehlstellen-muenchen-{year}"

# Stable local staging dirs — consistent paths matter for the watermark.
DIR_15MIN = Path("/tmp/munich_cycling/15min")
DIR_DAILY = Path("/tmp/munich_cycling/daily")

# ── Schemas ───────────────────────────────────────────────────────────────────
# Types match what polars.scan_csv infers from the raw files.
# The transform functions then parse dates and rename hyphenated columns.

SCHEMA_15MIN = {
    "datum": pl.Utf8,
    "uhrzeit_start": pl.Utf8,
    "uhrzeit_ende": pl.Utf8,
    "zaehlstelle": pl.Utf8,
    "richtung_1": pl.Int64,
    "richtung_2": pl.Int64,
    "gesamt": pl.Int64,
}

# "min-temp" / "max-temp" use hyphens — they are renamed to underscores in the transform.
SCHEMA_DAILY = {
    "datum": pl.Utf8,
    "uhrzeit_start": pl.Utf8,
    "uhrzeit_ende": pl.Utf8,
    "zaehlstelle": pl.Utf8,
    "richtung_1": pl.Int64,
    "richtung_2": pl.Int64,
    "gesamt": pl.Int64,
    "min-temp": pl.Float64,
    "max-temp": pl.Float64,
    "niederschlag": pl.Float64,
    "bewoelkung": pl.Int64,
    "sonnenstunden": pl.Float64,
}

# ── Pipelines ─────────────────────────────────────────────────────────────────


@incremental(
    source=str(DIR_15MIN),
    target=f"s3://{BUCKET}/delta/munich_cycling_15min",
    merge_on=None,  # append-only — watermark prevents reprocessing the same file
    file_format="csv",
    partition_by="year",
    reader_kwargs={"null_values": ["NA"]},
)
@schema(expect=SCHEMA_15MIN, on_extra="drop", evolution="cast")
def counts_15min(lf: pl.LazyFrame) -> pl.LazyFrame:
    parsed_datum = pl.coalesce(
        pl.col("datum").str.to_date("%Y.%m.%d", strict=False),
        pl.col("datum").str.to_date("%d.%m.%Y", strict=False),
    )
    return lf.with_columns(parsed_datum.alias("datum")).with_columns(
        pl.col("datum").dt.year().alias("year")
    )


@incremental(
    source=str(DIR_DAILY),
    target=f"s3://{BUCKET}/delta/munich_cycling_daily",
    merge_on=["datum", "zaehlstelle"],  # upsert — weather values may be revised retroactively
    file_format="csv",
    compact_every=6,  # compact after every 6 months of data
    partition_by="year",
    reader_kwargs={
        "null_values": ["NA"],
        # Force time columns to String — some files use dots (00.00) or 3-part
        # format (00:00:00) which Polars would otherwise infer as Float64 or Time.
        "schema_overrides": {"uhrzeit_start": pl.Utf8, "uhrzeit_ende": pl.Utf8},
    },
)
@schema(expect=SCHEMA_DAILY, on_extra="drop", evolution="merge")
def counts_daily(lf: pl.LazyFrame) -> pl.LazyFrame:
    parsed_datum = pl.coalesce(
        pl.col("datum").str.to_date("%Y.%m.%d", strict=False),
        pl.col("datum").str.to_date("%d.%m.%Y", strict=False),
    )
    return (
        lf.rename({"min-temp": "min_temp", "max-temp": "max_temp"})
        .with_columns(parsed_datum.alias("datum"))
        .with_columns(pl.col("datum").dt.year().alias("year"))
    )


# ── Fetch helpers ─────────────────────────────────────────────────────────────


def _normalize_csv(path: Path) -> None:
    """Fix known data-quality issues in the München cycling CSVs in-place.

    Handles files where the CSV was exported with a leading row-index column,
    producing a header like ``"\"\"","datum",...`` and data rows like
    ``"\"1\"","2023.12.01",...``.  The extra column is stripped so the file
    matches the expected schema.
    """
    text = path.read_text(encoding="utf-8")
    first_line = text.split("\n")[0]
    if not first_line.startswith('""'):
        return  # nothing to fix
    reader = csv.reader(io.StringIO(text))
    rows = [row[1:] for row in reader if row]  # drop leading index column
    # Fix dot-separated weather column names used in some exports
    rows[0] = [c.replace("min.temp", "min-temp").replace("max.temp", "max-temp") for c in rows[0]]
    out = io.StringIO()
    csv.writer(out).writerows(rows)
    path.write_text(out.getvalue(), encoding="utf-8")
    logger.info("Normalized (stripped index column): %s", path.name)


def _fetch_resources(year: int) -> tuple[list[dict], list[dict]]:
    """Return ``(resources_15min, resources_daily)`` for *year* from the CKAN API."""
    url = CKAN_API.format(year=year)
    req = urllib.request.Request(url, headers={"User-Agent": "data-warehousing-with-polars-demo"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())

    if not data.get("success"):
        raise RuntimeError(f"CKAN API returned an error for {year}: {data}")

    resources = data["result"]["resources"]
    r15 = [r for r in resources if "15 Minuten" in r.get("name", "")]
    rdaily = [r for r in resources if "Tageswerte" in r.get("name", "")]
    logger.info("%d: %d 15-min resources, %d daily resources.", year, len(r15), len(rdaily))
    return r15, rdaily


def _download(url: str, dest_dir: Path) -> Path | None:
    """Download *url* into *dest_dir* unless the file already exists.

    Returns ``None`` for corrected-data (*_korr*) files, which are skipped
    because they duplicate the original monthly files with a different schema.
    """
    filename = url.rstrip("/").split("/")[-1]
    if "_korr" in filename:
        logger.debug("Skipping corrected file: %s", filename)
        return None
    dest = dest_dir / filename
    if dest.exists():
        logger.info("Already present: %s", filename)
        return dest
    logger.info("Downloading %s …", url)
    req = urllib.request.Request(url, headers={"User-Agent": "data-warehousing-with-polars-demo"})
    with urllib.request.urlopen(req, timeout=60) as resp, dest.open("wb") as fh:
        fh.write(resp.read())
    _normalize_csv(dest)
    return dest


def fetch_all() -> None:
    """Download all available CSVs for 2022–present into local staging dirs."""
    DIR_15MIN.mkdir(parents=True, exist_ok=True)
    DIR_DAILY.mkdir(parents=True, exist_ok=True)

    for year in range(FIRST_YEAR, CURRENT_YEAR + 1):
        r15, rdaily = _fetch_resources(year)
        for r in r15:
            _download(r["url"], DIR_15MIN)
        for r in rdaily:
            _download(r["url"], DIR_DAILY)

    # Normalize any files that were downloaded before this fix was in place.
    for path in (*DIR_15MIN.glob("*.csv"), *DIR_DAILY.glob("*.csv")):
        _normalize_csv(path)


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    fetch_all()

    processed_15min = counts_15min.run()
    logger.info("15-min pipeline: %d new file(s) ingested.", len(processed_15min))

    processed_daily = counts_daily.run()
    logger.info("Daily pipeline: %d new file(s) ingested.", len(processed_daily))


if __name__ == "__main__":
    main()
