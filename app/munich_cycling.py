"""
Munich Raddauerzählstellen demo pipeline.

Fetches monthly cycling-counter CSVs from the Munich Open Data Portal and
loads them into Delta tables on S3 using data-warehousing-with-polars.

Two pipelines:
  - counts_15min  Append-only 15-minute interval counts per counting station.
  - counts_daily  Daily totals with weather data (temperature, precipitation,
                  cloud cover, sunshine hours). Upserts on (datum, zaehlstelle)
                  so weather values that are revised retroactively are corrected.

Each pipeline uses ``_CkanCyclingSource``, a custom :class:`Source` that:

  1. Queries the Munich Open Data CKAN API for CSV download URLs.
  2. Compares against the cursor (a sorted list of already-processed URLs).
  3. Downloads new files into a local staging directory, normalising known
     data-quality issues (leading index column, dot-separated column names).
  4. Returns a concatenated :class:`~polars.LazyFrame` of all new files.

The cursor grows with every processed URL, so each run only downloads files
that were not present when the pipeline last ran.

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
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import cast

import polars as pl
from data_warehousing_with_polars import incremental, schema
from data_warehousing_with_polars.incremental import Batch

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print("python-dotenv not found, skipping .env loading")


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

BUCKET = os.environ.get("MUNICH_CYCLING_BUCKET", "data-warehousing-with-polars")
FIRST_YEAR = 2022  # earliest year available on the portal
CURRENT_YEAR = datetime.now().year
CKAN_API = "https://opendata.muenchen.de/api/3/action/package_show?id=daten-der-raddauerzaehlstellen-muenchen-{year}"

# Stable local staging dirs — consistent paths matter for idempotent downloads.
DIR_15MIN = Path("/tmp/munich_cycling/15min")
DIR_DAILY = Path("/tmp/munich_cycling/daily")

# ── Schemas ───────────────────────────────────────────────────────────────────

SCHEMA_15MIN = {
    "datum": pl.Utf8,
    "uhrzeit_start": pl.Utf8,
    "uhrzeit_ende": pl.Utf8,
    "zaehlstelle": pl.Utf8,
    "richtung_1": pl.Int64,
    "richtung_2": pl.Int64,
    "gesamt": pl.Int64,
}

# "min-temp" / "max-temp" use hyphens — renamed to underscores in the transform.
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

# ── Download helpers ──────────────────────────────────────────────────────────


def _normalize_csv(path: Path) -> None:
    """Fix known data-quality issues in the München cycling CSVs in-place."""
    text = path.read_text(encoding="utf-8")
    first_line = text.split("\n")[0]
    if not first_line.startswith('""'):
        return
    reader = csv.reader(io.StringIO(text))
    rows = [row[1:] for row in reader if row]
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


# ── Custom Source implementation ──────────────────────────────────────────────


class _CkanCyclingSource:
    """Source that polls the Munich cycling CKAN API and downloads new CSVs.

    Cursor: a sorted list of all previously-processed download URLs.  On each
    ``poll``, URLs absent from the cursor are fetched and concatenated into a
    single LazyFrame.  The new cursor is the union of the old cursor and the
    newly downloaded URLs.

    Args:
        dest_dir:      Local directory for downloaded CSV files.
        resource_idx:  ``0`` for 15-minute files, ``1`` for daily files.
        reader_kwargs: Forwarded to :func:`polars.scan_csv`.
    """

    def __init__(
        self,
        dest_dir: Path,
        resource_idx: int,
        reader_kwargs: dict | None = None,
    ) -> None:
        self._dest_dir = dest_dir
        self._resource_idx = resource_idx
        self._reader_kwargs = reader_kwargs or {}

    def poll(self, since: object | None) -> Batch | None:
        processed: set[str] = set(cast(list[str], since)) if isinstance(since, list) else set()
        self._dest_dir.mkdir(parents=True, exist_ok=True)

        new_paths: list[str] = []
        new_urls: list[str] = []
        try:
            for year in range(FIRST_YEAR, CURRENT_YEAR + 1):
                for r in _fetch_resources(year)[self._resource_idx]:
                    url = r["url"]
                    if url not in processed:
                        path = _download(url, self._dest_dir)
                        if path is not None:
                            new_paths.append(str(path))
                            new_urls.append(url)
        except urllib.error.HTTPError as exc:
            if exc.code == 503:
                logger.warning("Portal returned 503 — treating as no new data.")
                return None
            raise

        if not new_paths:
            logger.info("No new cycling CSVs found.")
            return None

        logger.info("%d new cycling CSV(s) found.", len(new_paths))
        lf = pl.concat([pl.scan_csv(p, **self._reader_kwargs) for p in new_paths])
        return Batch(
            frame=lf,
            cursor=sorted(processed | set(new_urls)),
        )


# ── Pipelines ─────────────────────────────────────────────────────────────────


@incremental(
    source=_CkanCyclingSource(
        DIR_15MIN,
        resource_idx=0,
        reader_kwargs={"null_values": ["NA"]},
    ),
    target=f"s3://{BUCKET}/delta/munich_cycling_15min",
    merge_on=None,
    partition_by="year",
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
    source=_CkanCyclingSource(
        DIR_DAILY,
        resource_idx=1,
        reader_kwargs={
            "null_values": ["NA"],
            "schema_overrides": {"uhrzeit_start": pl.Utf8, "uhrzeit_ende": pl.Utf8},
        },
    ),
    target=f"s3://{BUCKET}/delta/munich_cycling_daily",
    merge_on=["datum", "zaehlstelle"],
    compact_every=6,
    partition_by="year",
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


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    processed_15min = counts_15min.run()
    logger.info("15-min pipeline: %d new file(s) ingested.", len(processed_15min))

    processed_daily = counts_daily.run()
    logger.info("Daily pipeline: %d new file(s) ingested.", len(processed_daily))


if __name__ == "__main__":
    main()
