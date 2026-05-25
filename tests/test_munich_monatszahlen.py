"""Tests for the munich_monatszahlen pipeline module."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------

# The app module lives under app/, which is not a package.  Add the repo root
# to sys.path so we can import it directly.
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from munich_monatszahlen import (  # noqa: E402
    _CKAN_SLUGS,
    _NATURAL_KEYS,
    _URLS,
    CKAN_API,
    SCHEMA_MONATSZAHLEN,
    _already_processed,
    _fetch_ckan_url,
    _ingest,
    _save_stamp,
    _transform,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_ROWS = [
    {
        "MONATSZAHL": "Gäste",
        "AUSPRAEGUNG": "insgesamt",
        "JAHR": 2023,
        "MONAT": "202301",
        "WERT": 1234.0,
        "VORJAHRESWERT": 1100.0,
        "VERAEND_VORMONAT_PROZENT": 2.5,
        "VERAEND_VORJAHRESMONAT_PROZENT": 12.2,
        "ZWOELF_MONATE_MITTELWERT": 1150.0,
    },
    {
        "MONATSZAHL": "Gäste",
        "AUSPRAEGUNG": "insgesamt",
        "JAHR": 2023,
        "MONAT": "202302",
        "WERT": 1400.0,
        "VORJAHRESWERT": 1200.0,
        "VERAEND_VORMONAT_PROZENT": 3.1,
        "VERAEND_VORJAHRESMONAT_PROZENT": 16.7,
        "ZWOELF_MONATE_MITTELWERT": 1160.0,
    },
    # Annual summary row — should be filtered out by _transform.
    {
        "MONATSZAHL": "Gäste",
        "AUSPRAEGUNG": "insgesamt",
        "JAHR": 2023,
        "MONAT": "Summe",
        "WERT": 12000.0,
        "VORJAHRESWERT": 10000.0,
        "VERAEND_VORMONAT_PROZENT": None,
        "VERAEND_VORJAHRESMONAT_PROZENT": 20.0,
        "ZWOELF_MONATE_MITTELWERT": 1000.0,
    },
    # Duplicate of first row — should be deduplicated.
    {
        "MONATSZAHL": "Gäste",
        "AUSPRAEGUNG": "insgesamt",
        "JAHR": 2023,
        "MONAT": "202301",
        "WERT": 1234.0,
        "VORJAHRESWERT": 1100.0,
        "VERAEND_VORMONAT_PROZENT": 2.5,
        "VERAEND_VORJAHRESMONAT_PROZENT": 12.2,
        "ZWOELF_MONATE_MITTELWERT": 1150.0,
    },
]


@pytest.fixture
def sample_df() -> pl.DataFrame:
    return pl.DataFrame(SAMPLE_ROWS, schema=SCHEMA_MONATSZAHLEN)


# ---------------------------------------------------------------------------
# Constants / config
# ---------------------------------------------------------------------------


def test_ckan_slugs_keys():
    """Tourism and labor entries must be present in _CKAN_SLUGS."""
    assert "tourism" in _CKAN_SLUGS
    assert "labor" in _CKAN_SLUGS


def test_ckan_slugs_values():
    """Slugs must match the known Munich Open Data package IDs."""
    assert _CKAN_SLUGS["tourism"] == "monatszahlen-tourismus"
    assert _CKAN_SLUGS["labor"] == "monatszahlen-arbeitsmarkt"


def test_original_urls_present():
    """The three original datasets must still have hardcoded URLs."""
    for key in ("accidents", "airport", "vehicles"):
        assert key in _URLS
        assert _URLS[key].startswith("https://opendata.muenchen.de/")


def test_ckan_api_template():
    """CKAN_API must contain the {package} placeholder."""
    assert "{package}" in CKAN_API


def test_natural_keys():
    """_NATURAL_KEYS must identify a row uniquely."""
    assert set(_NATURAL_KEYS) == {"MONATSZAHL", "AUSPRAEGUNG", "JAHR", "MONAT"}


# ---------------------------------------------------------------------------
# _transform
# ---------------------------------------------------------------------------


def test_transform_filters_summe_rows(sample_df):
    """Rows with MONAT == 'Summe' must be dropped."""
    result = _transform(sample_df, "test_source.csv")
    assert "Summe" not in result["MONAT"].to_list()


def test_transform_deduplicates(sample_df):
    """Duplicate natural-key rows must be collapsed to one."""
    result = _transform(sample_df, "test_source.csv")
    jan_rows = result.filter(pl.col("MONAT") == "202301")
    assert len(jan_rows) == 1


def test_transform_adds_date_column(sample_df):
    """Output must contain a 'date' column of type Date."""
    result = _transform(sample_df, "test_source.csv")
    assert "date" in result.columns
    assert result["date"].dtype == pl.Date


def test_transform_date_parsing(sample_df):
    """'202301' must parse to 2023-01-01."""
    import datetime as dt

    result = _transform(sample_df, "test_source.csv")
    jan = result.filter(pl.col("MONAT") == "202301")
    assert jan["date"][0] == dt.date(2023, 1, 1)


def test_transform_adds_source_file_column(sample_df):
    """Output must contain a '_source_file' column matching the argument."""
    result = _transform(sample_df, "/tmp/tourismus_202501.csv")
    assert result["_source_file"][0] == "/tmp/tourismus_202501.csv"


def test_transform_adds_ingested_at_column(sample_df):
    """Output must contain an '_ingested_at' column of Datetime type."""
    result = _transform(sample_df, "test_source.csv")
    assert "_ingested_at" in result.columns
    assert result["_ingested_at"].dtype in (
        pl.Datetime,
        pl.Datetime("us"),
        pl.Datetime("us", "UTC"),
    )


def test_transform_output_row_count(sample_df):
    """4 input rows → 2 output rows (Summe dropped, duplicate collapsed)."""
    result = _transform(sample_df, "test_source.csv")
    assert len(result) == 2


def test_transform_preserves_wert(sample_df):
    """WERT values must survive the transform unchanged."""
    result = _transform(sample_df, "test_source.csv").sort("MONAT")
    assert result["WERT"][0] == pytest.approx(1234.0)
    assert result["WERT"][1] == pytest.approx(1400.0)


# ---------------------------------------------------------------------------
# _already_processed / _save_stamp (integration with local Delta)
# ---------------------------------------------------------------------------


def test_already_processed_returns_false_when_no_table(tmp_path):
    """Return False when the watermark table does not yet exist."""
    store = str(tmp_path / "watermark")
    assert _already_processed(store, "202501") is False


def test_save_and_check_stamp(tmp_path):
    """A stamp saved via _save_stamp must be found by _already_processed."""
    store = str(tmp_path / "watermark")
    _save_stamp(store, "202501", rows=42)
    assert _already_processed(store, "202501") is True


def test_different_stamp_not_found(tmp_path):
    """A different stamp must not be found after saving an unrelated stamp."""
    store = str(tmp_path / "watermark")
    _save_stamp(store, "202501", rows=10)
    assert _already_processed(store, "202412") is False


def test_save_stamp_multiple_months(tmp_path):
    """Multiple distinct stamps can be stored and queried independently."""
    store = str(tmp_path / "watermark")
    _save_stamp(store, "202501", rows=1)
    _save_stamp(store, "202502", rows=2)
    assert _already_processed(store, "202501") is True
    assert _already_processed(store, "202502") is True
    assert _already_processed(store, "202503") is False


# ---------------------------------------------------------------------------
# _fetch_ckan_url
# ---------------------------------------------------------------------------


def _mock_urlopen(response_bytes: bytes):
    """Return a context manager mock that yields response_bytes."""
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.read.return_value = response_bytes
    return mock_resp


def test_fetch_ckan_url_returns_first_csv(tmp_path):
    """_fetch_ckan_url must return the URL of the first CSV resource."""
    import json

    payload = json.dumps(
        {
            "success": True,
            "result": {
                "resources": [
                    {"format": "CSV", "url": "https://example.com/tourismus.csv"},
                    {"format": "PDF", "url": "https://example.com/tourismus.pdf"},
                ]
            },
        }
    ).encode()

    with patch("urllib.request.urlopen", return_value=_mock_urlopen(payload)):
        url = _fetch_ckan_url("monatszahlen-tourismus")

    assert url == "https://example.com/tourismus.csv"


def test_fetch_ckan_url_case_insensitive_format(tmp_path):
    """Format comparison must be case-insensitive (e.g. 'csv' == 'CSV')."""
    import json

    payload = json.dumps(
        {
            "success": True,
            "result": {
                "resources": [
                    {"format": "csv", "url": "https://example.com/data.csv"},
                ]
            },
        }
    ).encode()

    with patch("urllib.request.urlopen", return_value=_mock_urlopen(payload)):
        url = _fetch_ckan_url("monatszahlen-arbeitsmarkt")

    assert url == "https://example.com/data.csv"


def test_fetch_ckan_url_raises_on_api_failure():
    """RuntimeError must be raised when the CKAN API reports success=False."""
    import json

    payload = json.dumps({"success": False, "error": {"message": "not found"}}).encode()

    with patch("urllib.request.urlopen", return_value=_mock_urlopen(payload)):
        with pytest.raises(RuntimeError, match="CKAN API error"):
            _fetch_ckan_url("nonexistent-package")


def test_fetch_ckan_url_raises_when_no_csv():
    """RuntimeError must be raised when no CSV resource is available."""
    import json

    payload = json.dumps(
        {
            "success": True,
            "result": {
                "resources": [
                    {"format": "PDF", "url": "https://example.com/doc.pdf"},
                ]
            },
        }
    ).encode()

    with patch("urllib.request.urlopen", return_value=_mock_urlopen(payload)):
        with pytest.raises(RuntimeError, match="No CSV resource"):
            _fetch_ckan_url("monatszahlen-tourismus")


# ---------------------------------------------------------------------------
# _ingest — smoke test with local Delta (no S3, no network)
# ---------------------------------------------------------------------------


def test_ingest_skips_when_already_processed(tmp_path, sample_df):
    """_ingest must return 0 without downloading when the stamp exists."""
    target = str(tmp_path / "delta_table")
    stamp = datetime.now().strftime("%Y%m")
    _save_stamp(target + "/.stamp", stamp, rows=99)

    # Patch _download so any accidental call raises.
    with patch("munich_monatszahlen._download", side_effect=AssertionError("should not download")):
        result = _ingest("accidents", target)

    assert result == 0


def test_ingest_processes_new_data(tmp_path, sample_df):
    """_ingest must write rows to Delta and save a watermark stamp."""
    target = str(tmp_path / "delta_table")
    csv_path = tmp_path / "test_data.csv"
    sample_df.write_csv(csv_path)

    with patch("munich_monatszahlen._download", return_value=csv_path):
        rows_written = _ingest("accidents", target)

    # 4 input rows → 2 after transform (Summe dropped, duplicate collapsed).
    assert rows_written == 2

    # Watermark must now prevent re-processing.
    stamp = datetime.now().strftime("%Y%m")
    assert _already_processed(target + "/.stamp", stamp) is True
