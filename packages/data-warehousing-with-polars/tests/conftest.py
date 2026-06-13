"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Generator

import psutil
import pytest


class _PeakRSSTracker:
    """Background-thread sampler that records the maximum RSS of the current process."""

    def __init__(self, interval_s: float = 0.005) -> None:
        self._proc = psutil.Process(os.getpid())
        self._interval = interval_s
        self._peak = self._proc.memory_info().rss
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> int:
        """Stop sampling and return peak RSS in bytes."""
        self._running = False
        if self._thread:
            self._thread.join()
        return self._peak

    def _loop(self) -> None:
        while self._running:
            rss = self._proc.memory_info().rss
            if rss > self._peak:
                self._peak = rss
            time.sleep(self._interval)


# Session-level list populated by the autouse fixture below.
_peak_rss_results: list[tuple[str, int]] = []


@pytest.fixture
def peak_rss_delta() -> Generator[Callable[[], float], None, None]:
    """Yield a callable that returns the peak RSS increase in MB since the fixture started.

    Use this in tests that need to assert memory is bounded::

        def test_something(tmp_path, peak_rss_delta):
            run_expensive_operation(tmp_path)
            assert peak_rss_delta() < 256
    """
    proc = psutil.Process(os.getpid())
    baseline = proc.memory_info().rss
    tracker = _PeakRSSTracker()
    tracker.start()
    yield lambda: (tracker._peak - baseline) / (1024 * 1024)
    tracker.stop()


class _RSSMeasurement:
    """RSS tracker that supports mid-test baseline reset for precise sub-measurement."""

    def __init__(self) -> None:
        self._proc = psutil.Process(os.getpid())
        self._tracker = _PeakRSSTracker()
        self._tracker.start()
        self._baseline = self._proc.memory_info().rss

    def reset(self) -> None:
        """Reset baseline and clear the peak to the current RSS.

        Call this after any setup work to measure only what follows.
        """
        current = self._proc.memory_info().rss
        self._baseline = current
        self._tracker._peak = current

    def delta_mb(self) -> float:
        """Return peak RSS increase in MB since the last :meth:`reset` (or fixture start)."""
        return (self._tracker._peak - self._baseline) / (1024 * 1024)

    def stop(self) -> None:
        self._tracker.stop()


@pytest.fixture
def peak_rss() -> Generator[_RSSMeasurement, None, None]:
    """Yield an :class:`_RSSMeasurement` that supports mid-test baseline reset.

    Use this when you need to measure only a sub-section of a test::

        def test_streaming(tmp_path, peak_rss):
            setup_expensive_state(tmp_path)   # not measured
            peak_rss.reset()                  # start measuring here
            run_operation_under_test(tmp_path)
            assert peak_rss.delta_mb() < 128
    """
    m = _RSSMeasurement()
    yield m
    m.stop()


@pytest.fixture(autouse=True)
def _track_peak_rss(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    tracker = _PeakRSSTracker()
    tracker.start()
    yield
    peak = tracker.stop()
    _peak_rss_results.append((request.node.nodeid, peak))


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter,
    exitstatus: int,
    config: pytest.Config,
) -> None:
    if not _peak_rss_results:
        return
    terminalreporter.write_sep("=", "peak RSS per test")
    for nodeid, peak_bytes in sorted(_peak_rss_results, key=lambda x: x[1], reverse=True):
        mb = peak_bytes / (1024 * 1024)
        terminalreporter.write_line(f"  {mb:6.1f} MB  {nodeid}")


@pytest.fixture(autouse=True)
def python_dotenv() -> Generator[None, None, None]:
    """Load environment variables from .env before tests run."""
    from dotenv import load_dotenv

    load_dotenv()
    yield
