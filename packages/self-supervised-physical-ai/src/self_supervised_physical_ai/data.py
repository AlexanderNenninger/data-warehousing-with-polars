"""Datasets for LeJEPA training on industrial sensor windows."""

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from self_supervised_physical_ai.config import HashableBaseModel


class SyntheticSEFCDataset(Dataset):
    """Fake multivariate sensor windows with a few injected 'operating
    regimes' (different frequency/amplitude combos), so you can sanity-check
    that discovered clusters in embedding space correspond to something
    before pointing this at real FactoryNet data."""

    def __init__(
        self,
        n_windows: int,
        n_channels: int,
        window_len: int,
        sample_rate_hz: float = 100.0,
    ):
        self.n_windows = n_windows
        self.n_channels = n_channels
        self.window_len = window_len
        self.sample_rate_hz = sample_rate_hz
        self.regimes = torch.randint(0, 3, (n_windows,))  # 3 synthetic "operating states"

    def __len__(self):
        return self.n_windows

    def __getitem__(self, index):
        regime = int(self.regimes[index].item())
        t = torch.arange(self.window_len).float() / self.sample_rate_hz
        freq = [0.5, 2.0, 5.0][regime]
        amp = [1.0, 0.5, 2.0][regime]
        base = amp * torch.sin(2 * torch.pi * freq * t)
        x = base.unsqueeze(0).repeat(self.n_channels, 1)
        x = x + 0.1 * torch.randn_like(x)
        return {
            "x": x,  # (n_channels, window_len)
            "sample_rate_hz": torch.tensor(self.sample_rate_hz),
            "regime": regime,  # ground-truth label, for validating clusters later
        }


class FactoryNetWindowSpec(HashableBaseModel):
    """Which columns to window over, and how.

    `channel_columns` and `metadata_columns` are per-source — FactoryNet's files
    don't share a schema, so pick these via
    `factorynet.select_channel_columns`/`factorynet.available_metadata_columns`
    against the schema of the specific file(s) you loaded.
    """

    window_len: int
    stride: int
    channel_columns: tuple[str, ...]
    metadata_columns: tuple[str, ...] = ()


class FactoryNetWindowDataset(Dataset):
    """Sliding-window dataset over FactoryNet episodes.

    Takes an eagerly-loaded `pl.DataFrame` (already scoped to one source family,
    with an `episode_id` + `time_s` column plus whatever `spec` names) and slices
    fixed-length windows out of it. Windows never cross an episode boundary —
    episodes are discrete task executions with their own start/end semantics, so
    the tail that doesn't fill a full window is dropped rather than padded or
    slid into the next episode.

    `sample_rate_hz` is derived per-window from `time_s` (FactoryNet doesn't carry
    a separate sample-rate field), so it's allowed to vary slightly between
    windows rather than being assumed constant for the whole episode.
    """

    def __init__(self, df: pl.DataFrame, spec: FactoryNetWindowSpec):
        self.spec = spec
        self._episodes: dict[str, pl.DataFrame] = {
            str(episode_id): group
            for (episode_id,), group in df.group_by("episode_id", maintain_order=True)
        }
        self._index: list[tuple[str, int]] = [
            (episode_id, start)
            for episode_id, group in self._episodes.items()
            for start in range(0, group.height - spec.window_len + 1, spec.stride)
        ]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index):
        episode_id, start = self._index[index]
        window = self._episodes[episode_id].slice(start, self.spec.window_len)

        channels = window.select(list(self.spec.channel_columns)).to_numpy()
        x = torch.from_numpy(channels.copy()).float().T
        dt = np.diff(window["time_s"].to_numpy())
        dt_median = float(np.median(dt)) if len(dt) else 1.0
        sample_rate_hz = 1.0 / dt_median if dt_median > 0 else 1.0

        item = {
            "x": x,  # (n_channels, window_len)
            "sample_rate_hz": torch.tensor(sample_rate_hz, dtype=torch.float32),
            "episode_id": episode_id,
        }
        for col in self.spec.metadata_columns:
            item[col] = window[col][0]
        return item
