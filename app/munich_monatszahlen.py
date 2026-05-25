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
in-place every month.  Because the authoritative source is always the latest
complete file, the correct write strategy is *overwrite* rather than *merge*:
each monthly run downloads the file, transforms and deduplicates the data, and
atomically replaces the Delta table.  A lightweight month-stamp watermark stored
alongside the table prevents duplicate processing within the same calendar month.

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
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
from deltalake import write_deltalake

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
    return (
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


# ── Watermark helpers ─────────────────────────────────────────────────────────


def _already_processed(watermark_store: str, stamp: str) -> bool:
    """Return True if *stamp* (YYYYMM) is recorded in the watermark table."""
    try:
        stamps = (
            pl.scan_delta(watermark_store).select("stamp").collect()["stamp"].to_list()
        )
        return stamp in stamps
    except Exception:
        return False


def _save_stamp(watermark_store: str, stamp: str, rows: int) -> None:
    """Append *stamp* with metadata to the watermark table."""
    wm = pl.DataFrame({
        "stamp": [stamp],
        "written_at": [datetime.now(timezone.utc)],
        "rows": [rows],
    })
    write_deltalake(watermark_store, wm.to_arrow(), mode="append")


# ── Full-history ingest ───────────────────────────────────────────────────────


def _ingest(name: str, target: str) -> int:
    """Download, transform, and atomically overwrite the Delta table for *name*.

    Because the source is a full-history file replaced monthly, the correct
    write strategy is overwrite rather than merge: the latest published CSV is
    the authoritative version of the complete dataset.

    Returns the number of rows written, or 0 if this month's data was already
    ingested.
    """
    stamp = datetime.now().strftime("%Y%m")
    watermark_store = target + "/.stamp"

    if _already_processed(watermark_store, stamp):
        logger.info("[%s] Already processed for %s — skipping.", name, stamp)
        return 0

    csv_path = _download(name)

    df_raw = pl.read_csv(str(csv_path), null_values=["NA"], quote_char='"')
    logger.info("[%s] Raw CSV: %d rows.", name, len(df_raw))

    df = _transform(df_raw, str(csv_path))
    logger.info("[%s] After transform: %d rows.", name, len(df))

    write_deltalake(target, df.to_arrow(), mode="overwrite")
    _save_stamp(watermark_store, stamp, len(df))

    logger.info("[%s] Overwrite complete — %d rows written.", name, len(df))
    return len(df)


# ── Download helper ───────────────────────────────────────────────────────────


def _fetch_ckan_url(package_slug: str) -> str:
    """Return the first CSV download URL from the Munich Open Data CKAN API.

    Used for datasets whose filenames change with each monthly update so that
    the pipeline always downloads the current version without requiring manual
    URL updates.
    """
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
    """Download the named Monatszahlen CSV into a date-stamped staging file.

    URL resolution order:
    1. ``_URLS`` — stable hardcoded URLs for datasets whose filenames never change.
    2. ``_CKAN_SLUGS`` — CKAN API lookup for datasets with date-stamped filenames.
    """
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


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    for name, suffix in [
        ("accidents", "munich_accidents"),
        ("airport",   "munich_airport"),
        ("vehicles",  "munich_vehicles"),
        ("tourism",   "munich_tourism"),
        ("labor",     "munich_labor"),
    ]:
        target = f"s3://{BUCKET}/delta/{suffix}"
        n = _ingest(name, target)
        logger.info("%s: %d rows ingested.", name, n)


if __name__ == "__main__":
    main()


