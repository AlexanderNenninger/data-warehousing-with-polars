"""
Munich Monatszahlen demo pipelines.

Ingests five monthly statistical datasets published by the Statistisches Amt
München into Delta Lake tables on S3.

Five pipelines (all use the same "Monatszahlen Monitoring" CSV schema):

  - accidents   Monatszahlen Verkehrsunfälle — monthly road-accident counts
                (total, with injuries, alcohol-related, hit-and-run).

  - airport     Monatszahlen Flugverkehr — monthly Munich Airport passenger and
                flight-movement statistics.

  - vehicles    Monatszahlen KFZ-Bestand — monthly vehicle fleet by fuel type,
                useful for tracking EV adoption in Munich.

  - tourism     Monatszahlen Tourismus — monthly hotel guests and overnight stays,
                showing the COVID collapse and recovery.

  - labor       Monatszahlen Arbeitsmarkt — monthly unemployment figures and
                open job postings for Munich.

Each source is a **single full-history CSV** that the data portal replaces
in-place every month.  The pipeline uses :func:`from_query` with a YYYYMM
stamp cursor: the fetch function downloads the file and returns ``None`` if the
current month has already been processed.  :func:`incremental` stores the stamp
in its watermark table; the next run compares it to avoid duplicate downloads.

The three original datasets (accidents, airport, vehicles) use stable hardcoded
download URLs.  The two newer datasets (tourism, labor) look up their current
download URL via the Munich Open Data CKAN API so that date-stamped filenames
are handled automatically.

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

import json
import logging
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import polars as pl
from data_warehousing_with_polars import from_query, incremental

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print("python-dotenv not found, skipping .env loading")


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

BUCKET = os.environ.get("MUNICH_CYCLING_BUCKET", "data-warehousing-with-polars")

# Datasets with stable hardcoded download URLs.
_URLS: dict[str, str] = {
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

# Datasets whose filenames change on every portal update — discovered via CKAN API.
_CKAN_SLUGS: dict[str, str] = {
    "tourism": "monatszahlen-tourismus",
    "labor": "monatszahlen-arbeitsmarkt",
}

CKAN_API = "https://opendata.muenchen.de/api/3/action/package_show?id={package}"

_STAGING = Path("/tmp/munich_monatszahlen")

# ── Shared schema ─────────────────────────────────────────────────────────────

SCHEMA_MONATSZAHLEN = {
    "MONATSZAHL": pl.Utf8,
    "AUSPRAEGUNG": pl.Utf8,
    "JAHR": pl.Int64,
    "MONAT": pl.Utf8,
    "WERT": pl.Float64,
    "VORJAHRESWERT": pl.Float64,
    "VERAEND_VORMONAT_PROZENT": pl.Float64,
    "VERAEND_VORJAHRESMONAT_PROZENT": pl.Float64,
    "ZWOELF_MONATE_MITTELWERT": pl.Float64,
}

_NATURAL_KEYS = ["MONATSZAHL", "AUSPRAEGUNG", "JAHR", "MONAT"]


# ── Transform ─────────────────────────────────────────────────────────────────


def _transform(df: pl.DataFrame, source_path: str) -> pl.DataFrame:
    """Filter annual summary rows, parse MONAT into a date, and deduplicate.

    The portal occasionally publishes duplicate rows; deduplication on the
    natural keys ensures the Delta table stays clean regardless.
    """
    now = datetime.now(timezone.utc)
    result = (
        df.lazy()
        .filter(pl.col("MONAT") != "Summe")
        .with_columns(
            pl.col("MONAT").str.strptime(pl.Date, "%Y%m", strict=False).alias("date"),
            pl.lit(source_path).alias("_source_file"),
            pl.lit(now).alias("_ingested_at"),
        )
        .unique(subset=_NATURAL_KEYS, keep="last", maintain_order=False)
        .collect()
    )
    return cast(pl.DataFrame, result)


# ── Download helper ───────────────────────────────────────────────────────────


def _fetch_ckan_url(package_slug: str) -> str:
    """Return the first CSV download URL from the Munich Open Data CKAN API."""
    url = CKAN_API.format(package=package_slug)
    req = urllib.request.Request(url, headers={"User-Agent": "data-warehousing-with-polars-demo"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    if not data.get("success"):
        raise RuntimeError(f"CKAN API error for '{package_slug}': {data}")
    resources = data["result"]["resources"]
    csv_resources = [r for r in resources if r.get("format", "").upper() == "CSV"]
    if not csv_resources:
        raise RuntimeError(f"No CSV resource found for package '{package_slug}'")
    return csv_resources[0]["url"]


def _download(name: str) -> Path:
    """Download the named Monatszahlen CSV into a date-stamped staging file."""
    dest_dir = _STAGING / name
    dest_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m")
    dest = dest_dir / f"{name}_{stamp}.csv"
    if dest.exists():
        logger.info("[%s] Already downloaded: %s", name, dest.name)
        return dest

    if name in _URLS:
        url = _URLS[name]
    elif name in _CKAN_SLUGS:
        url = _fetch_ckan_url(_CKAN_SLUGS[name])
    else:
        raise ValueError(f"Unknown dataset name: '{name}'")

    logger.info("[%s] Downloading %s …", name, url)
    req = urllib.request.Request(url, headers={"User-Agent": "data-warehousing-with-polars-demo"})
    with urllib.request.urlopen(req, timeout=60) as resp, dest.open("wb") as fh:
        fh.write(resp.read())
    logger.info("[%s] Saved → %s", name, dest.name)
    return dest


# ── Source factories ──────────────────────────────────────────────────────────


def _make_source(name: str):
    """Return a ``from_query`` source for the named Monatszahlen dataset.

    The cursor is a YYYYMM stamp.  ``poll`` returns ``None`` when the current
    month has already been processed, otherwise downloads, transforms, and
    returns the full-history LazyFrame with a ``_stamp`` column that drives the
    cursor.  The pipeline transform drops ``_stamp`` before writing.
    """

    def _fetch(since: object | None) -> pl.LazyFrame | None:
        stamp = datetime.now().strftime("%Y%m")
        if since == stamp:
            logger.info("[%s] Already processed for %s — skipping.", name, stamp)
            return None
        try:
            csv_path = _download(name)
        except urllib.error.HTTPError as exc:
            if exc.code == 503:
                logger.warning("[%s] Portal returned 503 — treating as no new data.", name)
                return None
            raise
        df_raw = pl.read_csv(
            str(csv_path),
            schema_overrides=SCHEMA_MONATSZAHLEN,
            null_values=["NA"],
            quote_char='"',
        )
        logger.info("[%s] Raw CSV: %d rows.", name, len(df_raw))
        df = _transform(df_raw, str(csv_path))
        logger.info("[%s] After transform: %d rows.", name, len(df))
        return df.with_columns(pl.lit(stamp).alias("_stamp")).lazy()

    return from_query(_fetch, cursor_on="_stamp")


# ── Pipelines ─────────────────────────────────────────────────────────────────


@incremental(
    source=_make_source("accidents"),
    target=f"s3://{BUCKET}/delta/munich_accidents",
    history_target=f"s3://{BUCKET}/delta/munich_accidents_history",
    merge_on=_NATURAL_KEYS,
    scd_type=4,
)
def accidents(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.drop("_stamp")


@incremental(
    source=_make_source("airport"),
    target=f"s3://{BUCKET}/delta/munich_airport",
    history_target=f"s3://{BUCKET}/delta/munich_airport_history",
    merge_on=_NATURAL_KEYS,
    scd_type=4,
)
def airport(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.drop("_stamp")


@incremental(
    source=_make_source("vehicles"),
    target=f"s3://{BUCKET}/delta/munich_vehicles",
    history_target=f"s3://{BUCKET}/delta/munich_vehicles_history",
    merge_on=_NATURAL_KEYS,
    scd_type=4,
)
def vehicles(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.drop("_stamp")


@incremental(
    source=_make_source("tourism"),
    target=f"s3://{BUCKET}/delta/munich_tourism",
    history_target=f"s3://{BUCKET}/delta/munich_tourism_history",
    merge_on=_NATURAL_KEYS,
    scd_type=4,
)
def tourism(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.drop("_stamp")


@incremental(
    source=_make_source("labor"),
    target=f"s3://{BUCKET}/delta/munich_labor",
    history_target=f"s3://{BUCKET}/delta/munich_labor_history",
    merge_on=_NATURAL_KEYS,
    scd_type=4,
)
def labor(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.drop("_stamp")


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    for pipeline in [accidents, airport, vehicles, tourism, labor]:
        result = pipeline.run()
        if result:
            logger.info("%s: ingested (cursor=%s).", pipeline.__name__, result[0])
        else:
            logger.info("%s: already up to date.", pipeline.__name__)


if __name__ == "__main__":
    main()
