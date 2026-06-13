"""Smoke tests: confirm the compiled plugin loads and core expressions run."""

import polars as pl
import polars_list_ext as ple


def test_version_is_exposed():
    assert isinstance(ple._internal.__version__, str)


def test_apply_fft_returns_expected_bin_count():
    # A real FFT of a length-N signal yields N // 2 + 1 magnitude bins.
    n = 16
    df = pl.DataFrame({"sig": [[float(i % 4) for i in range(n)]]})
    out = df.with_columns(ple.apply_fft("sig", sample_rate=n).alias("fft"))
    assert out["fft"].list.len().to_list() == [n // 2 + 1]


def test_apply_fft_is_lazy_compatible():
    n = 8
    lf = pl.LazyFrame({"sig": [[1.0] * n, [0.0] * n]})
    out = lf.with_columns(ple.apply_fft("sig", sample_rate=n).alias("fft")).collect()
    assert out.height == 2
    assert out["fft"].list.len().to_list() == [n // 2 + 1, n // 2 + 1]
