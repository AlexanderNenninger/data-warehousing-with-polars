"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import os
import threading
import time

import psutil


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
