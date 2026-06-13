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
    _fetch_ckan_url,
    _make_source,
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
# _make_source — cursor / poll behaviour
# ---------------------------------------------------------------------------


def test_make_source_skips_current_stamp():
    """poll returns None when cursor matches the current YYYYMM stamp."""
    stamp = datetime.now().strftime("%Y%m")
    src = _make_source("accidents")
    assert src.poll(stamp) is None


def test_make_source_returns_batch_for_old_stamp(tmp_path, sample_df):
    """poll returns a Batch with a _stamp column when the cursor is stale."""
    csv_path = tmp_path / "test.csv"
    sample_df.write_csv(csv_path)
    src = _make_source("accidents")
    with patch("munich_monatszahlen._download", return_value=csv_path):
        batch = src.poll("202001")
    assert batch is not None
    stamp = datetime.now().strftime("%Y%m")
    assert batch.cursor == stamp
    assert "_stamp" in batch.frame.collect().columns


def test_make_source_returns_batch_on_first_run(tmp_path, sample_df):
    """poll returns a Batch when since=None (initial load)."""
    csv_path = tmp_path / "test.csv"
    sample_df.write_csv(csv_path)
    src = _make_source("accidents")
    with patch("munich_monatszahlen._download", return_value=csv_path):
        batch = src.poll(None)
    assert batch is not None
    assert batch.cursor == datetime.now().strftime("%Y%m")


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
# Pipeline integration — smoke test with local Delta (no S3, no network)
# ---------------------------------------------------------------------------


def test_pipeline_processes_and_skips_on_rerun(tmp_path, sample_df):
    """Pipeline writes data on first run and skips on the second (same month)."""
    from data_warehousing_with_polars import incremental

    csv_path = tmp_path / "test_data.csv"
    sample_df.write_csv(csv_path)
    target = str(tmp_path / "delta")

    @incremental(
        source=_make_source("accidents"),
        target=target,
        merge_on=_NATURAL_KEYS,
    )
    def test_pipe(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.drop("_stamp")

    with patch("munich_monatszahlen._download", return_value=csv_path):
        result1 = test_pipe.run()

    assert len(result1) > 0

    result2 = test_pipe.run()
    assert result2 == []
