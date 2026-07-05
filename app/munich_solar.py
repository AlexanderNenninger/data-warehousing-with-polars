"""
Munich solar + battery storage demo pipelines.

Ingests two independent real-world time series that feed the "solar + battery
economics" example (``examples/munich-solar.qmd``):

  - spot_price        German day-ahead electricity price (SMARD / Bundesnetzagentur),
                      hourly, DE-LU market area. This is the reference price a
                      dynamic electricity tariff is indexed to.
  - solar_irradiance  Hourly global radiation (DWD / Deutscher Wetterdienst),
                      used to model rooftop PV generation.

Munich itself has no DWD radiation-measuring station, so the nearest one —
Weihenstephan-Dürnast (station 05404, ~30 km north, near Freising/Munich
Airport) — is used as a proxy for Munich weather.

``spot_price`` uses a custom :class:`Source` (``_SmardSource``) that polls
SMARD's weekly JSON chunks. ``solar_irradiance`` uses :func:`from_frame`,
since DWD publishes the whole station history as a single file with no
time-range API — a full re-fetch every run, deduplicated on merge.

Both pipelines are bounded to data from ``FIRST_DATE`` onward so the demo
tables stay a few years of hourly data rather than SMARD/DWD's full history
(2015-present and 1961-present respectively).

Data:    SMARD (Bundesnetzagentur) — https://www.smard.de
         DWD Climate Data Center — https://opendata.dwd.de
License: Datenlizenz Deutschland Namensnennung 2.0
         https://www.govdata.de/dl-de/by-2-0

Required environment variables:
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_REGION               (or AWS_ENDPOINT_URL for non-AWS S3)
    MUNICH_CYCLING_BUCKET    e.g. "my-bucket"
"""

from __future__ import annotations

import io
import json
import logging
import os
import urllib.request
import zipfile
from datetime import date, datetime, timezone

import polars as pl
from data_warehousing_with_polars import Batch, from_frame, incremental, schema

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print("python-dotenv not found, skipping .env loading")


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_USER_AGENT = {"User-Agent": "data-warehousing-with-polars-demo"}

# ── Config ────────────────────────────────────────────────────────────────────

BUCKET = os.environ.get("MUNICH_CYCLING_BUCKET", "data-warehousing-with-polars")

# Bounds both pipelines to a few recent years instead of SMARD's full 2015-present
# / DWD's full 1961-present history.
FIRST_DATE = date(2023, 1, 1)
_FIRST_DATETIME = datetime(FIRST_DATE.year, FIRST_DATE.month, FIRST_DATE.day)
_FIRST_TS_MS = int(_FIRST_DATETIME.replace(tzinfo=timezone.utc).timestamp() * 1000)

# ── SMARD day-ahead price ────────────────────────────────────────────────────

_SMARD_FILTER = 4169  # Day-ahead price, DE-LU market area
_SMARD_REGION = "DE-LU"
_SMARD_RESOLUTION = "hour"
_SMARD_BASE = "https://www.smard.de/app/chart_data"


class _SmardSource:
    """Source that polls the SMARD day-ahead price API for new weekly chunks.

    SMARD publishes price data in ~weekly JSON chunks, indexed by an
    ``index_{resolution}.json`` file listing chunk timestamps. Cursor: the
    timestamp of the *second-most-recent* chunk fetched, not the most recent —
    so the freshest chunk is always re-fetched on the next run to pick up
    SMARD's late corrections to near-real-time prices. ``merge_on`` on the
    pipeline absorbs the resulting overlap.
    """

    def __init__(self, filter_id: int, region: str, resolution: str) -> None:
        self._filter_id = filter_id
        self._region = region
        self._resolution = resolution

    def _index_url(self) -> str:
        return f"{_SMARD_BASE}/{self._filter_id}/{self._region}/index_{self._resolution}.json"

    def _chunk_url(self, ts: int) -> str:
        return (
            f"{_SMARD_BASE}/{self._filter_id}/{self._region}"
            f"/{self._filter_id}_{self._region}_{self._resolution}_{ts}.json"
        )

    def _get_json(self, url: str) -> dict:
        req = urllib.request.Request(url, headers=_USER_AGENT)
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())

    def poll(self, since: object | None) -> Batch | None:
        index = self._get_json(self._index_url())
        timestamps = sorted(t for t in index["timestamps"] if t >= _FIRST_TS_MS)
        if since is not None:
            timestamps = [t for t in timestamps if t >= since]
        if not timestamps:
            logger.info("SMARD: no new chunks.")
            return None

        ts_ms: list[int] = []
        prices: list[float | None] = []
        for ts in timestamps:
            payload = self._get_json(self._chunk_url(ts))
            for point_ts, price in payload["series"]:
                ts_ms.append(point_ts)
                prices.append(price)

        lf = (
            pl.DataFrame({"ts_ms": ts_ms, "price_eur_mwh": prices})
            .lazy()
            .filter(pl.col("price_eur_mwh").is_not_null())
            .select(
                pl.col("ts_ms")
                .cast(pl.Datetime("ms"))
                .dt.replace_time_zone("UTC")
                .alias("timestamp"),
                pl.col("price_eur_mwh").cast(pl.Float64),
            )
        )
        # Lag the cursor by one chunk so the freshest chunk is always re-checked.
        cursor = timestamps[-2] if len(timestamps) >= 2 else timestamps[-1]
        logger.info("SMARD: %d new chunk(s), cursor=%s.", len(timestamps), cursor)
        return Batch(frame=lf, cursor=cursor)


# ── DWD hourly solar irradiance ──────────────────────────────────────────────

_DWD_STATION_ID = "05404"  # Weihenstephan-Dürnast — nearest DWD radiation station to Munich
_DWD_ZIP_URL = (
    "https://opendata.dwd.de/climate_environment/CDC/observations_germany/"
    f"climate/hourly/solar/stundenwerte_ST_{_DWD_STATION_ID}_row.zip"
)


def _fetch_dwd_solar() -> pl.LazyFrame:
    """Download and parse the full Weihenstephan-Dürnast hourly solar record.

    DWD ships the entire station history as one file with no time-range API,
    so this always fetches everything and relies on ``merge_on`` to dedupe.

    ``MESS_DATUM_WOZ`` (not ``MESS_DATUM``) is used as the hour timestamp: it's
    the round-hour field in this dataset, which is what matters for joining
    against SMARD's hourly prices — the true-local-solar-time vs. UTC-clock
    distinction between the two DWD date columns is a few minutes and
    immaterial at hourly economics granularity.

    ``FG_LBERG`` (hourly global radiation sum, J/cm²) is converted to
    kWh/m² by dividing by 360 (1 kWh/m² = 3.6 MJ/m² = 360 J/cm²). ``-999`` is
    DWD's missing-value marker.
    """
    req = urllib.request.Request(_DWD_ZIP_URL, headers=_USER_AGENT)
    with urllib.request.urlopen(req, timeout=60) as resp:
        archive_bytes = resp.read()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
        product_name = next(n for n in zf.namelist() if n.startswith("produkt_st_stunde_"))
        with zf.open(product_name) as fh:
            product_bytes = fh.read()

    raw = pl.read_csv(io.BytesIO(product_bytes), separator=";", infer_schema=False)

    fg = pl.col("FG_LBERG").str.strip_chars().cast(pl.Float64)
    sd = pl.col("SD_LBERG").str.strip_chars().cast(pl.Float64)

    return (
        raw.lazy()
        .select(
            pl.col("MESS_DATUM_WOZ")
            .str.strip_chars()
            .str.strptime(pl.Datetime, "%Y%m%d%H:%M")
            .alias("datetime_utc"),
            pl.when(fg == -999).then(None).otherwise(fg / 360.0).alias("ghi_kwh_per_m2"),
            pl.when(sd == -999).then(None).otherwise(sd).alias("sunshine_min"),
        )
        .filter(pl.col("datetime_utc") >= _FIRST_DATETIME)
    )


# ── Pipelines ─────────────────────────────────────────────────────────────────


@incremental(
    source=_SmardSource(_SMARD_FILTER, _SMARD_REGION, _SMARD_RESOLUTION),
    target=f"s3://{BUCKET}/delta/munich_solar_spot_price",
    merge_on=["timestamp"],
)
@schema(
    expect={"timestamp": pl.Datetime("ms", "UTC"), "price_eur_mwh": pl.Float64},
    on_extra="drop",
    evolution="cast",
)
def spot_price(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf


@incremental(
    source=from_frame(_fetch_dwd_solar),
    target=f"s3://{BUCKET}/delta/munich_solar_irradiance",
    merge_on=["datetime_utc"],
)
@schema(
    expect={
        "datetime_utc": pl.Datetime("us"),
        "ghi_kwh_per_m2": pl.Float64,
        "sunshine_min": pl.Float64,
    },
    on_extra="drop",
    evolution="cast",
)
def solar_irradiance(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    processed_price = spot_price.run()
    logger.info(
        "Spot price: %s",
        f"{len(processed_price)} new chunk(s)." if processed_price else "up to date.",
    )

    processed_irradiance = solar_irradiance.run()
    logger.info("Solar irradiance: %s", "refreshed." if processed_irradiance else "up to date.")


if __name__ == "__main__":
    main()
