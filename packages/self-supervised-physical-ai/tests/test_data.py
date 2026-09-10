"""Tests for datasets."""

import numpy as np
import polars as pl
import pytest
from self_supervised_physical_ai import (
    FactoryNetWindowDataset,
    FactoryNetWindowSpec,
    SyntheticSEFCDataset,
)


def test_synthetic_sefc_dataset_shapes():
    ds = SyntheticSEFCDataset(n_windows=5, n_channels=3, window_len=64, sample_rate_hz=50.0)
    assert len(ds) == 5

    sample = ds[0]
    assert sample["x"].shape == (3, 64)
    assert sample["sample_rate_hz"].item() == 50.0
    assert sample["regime"] in (0, 1, 2)


def _fake_episode(episode_id: str, n_rows: int, hz: float, offset: float) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "episode_id": [episode_id] * n_rows,
            "time_s": (np.arange(n_rows) / hz).tolist(),
            "setpoint_pos_0": (np.arange(n_rows) + offset).tolist(),
            "feedback_pos_0": (np.arange(n_rows) + offset + 0.1).tolist(),
            "machine_type": ["UR3"] * n_rows,
        }
    )


def test_factorynet_window_dataset_respects_episode_boundaries():
    df = pl.concat(
        [
            _fake_episode("ep-a", n_rows=25, hz=100.0, offset=0.0),
            _fake_episode("ep-b", n_rows=10, hz=100.0, offset=1000.0),
        ]
    )
    spec = FactoryNetWindowSpec(
        window_len=10,
        stride=10,
        channel_columns=("setpoint_pos_0", "feedback_pos_0"),
        metadata_columns=("machine_type",),
    )
    ds = FactoryNetWindowDataset(df, spec)

    # ep-a (25 rows) yields 2 full windows of 10 at stride 10; ep-b (10 rows) yields 1.
    # No window may straddle the two episodes.
    assert len(ds) == 3

    for i in range(len(ds)):
        item = ds[i]
        assert item["x"].shape == (2, 10)
        assert item["machine_type"] == "UR3"
        # channel values must all come from a single episode (offsets are 0 vs 1000).
        assert (item["x"] < 500).all() or (item["x"] >= 500).all()


def test_factorynet_window_dataset_sample_rate_from_time_s():
    df = _fake_episode("ep-a", n_rows=20, hz=200.0, offset=0.0)
    spec = FactoryNetWindowSpec(window_len=10, stride=10, channel_columns=("setpoint_pos_0",))
    ds = FactoryNetWindowDataset(df, spec)

    item = ds[0]
    assert item["sample_rate_hz"].item() == pytest.approx(200.0, rel=1e-3)
