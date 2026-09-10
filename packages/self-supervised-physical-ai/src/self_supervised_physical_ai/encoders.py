"""Patch embedding, positional encoding, and the LeJEPA transformer encoder."""

import math

import torch
from torch import Tensor, nn

from self_supervised_physical_ai.config import LeJEPAConfig


def time_positional_encoding(t: Tensor, d_model: int, max_period: float = 100.0) -> Tensor:
    """Continuous-time (not index-based) positional encoding.

    Composes across different source sampling rates, e.g. mixing 100/500/1000
    Hz accelerometer channels in the same window.

    Args:
        t: (batch, n_patches) float tensor of seconds-since-window-start for
            each patch (e.g. patch center time). Same window can mix patches
            from different source sampling rates as long as `t` reflects true
            elapsed time.
        d_model: embedding dimension (must be even).
        max_period: longest wavelength in the encoding, in seconds.

    Returns:
        (batch, n_patches, d_model)
    """
    assert d_model % 2 == 0
    half = d_model // 2
    device = t.device
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=device).float() / half)
    args = t.unsqueeze(-1) * freqs  # (batch, n_patches, half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


def patch_times(window_len: int, patch_len: int, sample_rate_hz: Tensor) -> Tensor:
    """Patch center times in seconds.

    Args:
        window_len: timesteps per window.
        patch_len: timesteps per patch.
        sample_rate_hz: (batch,) Hz of the source each window came from.

    Returns:
        (batch, n_patches)
    """
    n_patches = window_len // patch_len
    idx = torch.arange(n_patches, device=sample_rate_hz.device).float()
    patch_center_step = idx * patch_len + patch_len / 2  # in samples
    dt = 1.0 / sample_rate_hz.unsqueeze(-1)  # (batch, 1) seconds/sample
    return patch_center_step.unsqueeze(0) * dt  # (batch, n_patches)


def patchify(x: Tensor, patch_len: int) -> Tensor:
    """x: (batch, n_channels, window_len) -> (batch, n_patches, n_channels*patch_len)"""
    b, c, t = x.shape
    assert t % patch_len == 0
    n_patches = t // patch_len
    x = x.view(b, c, n_patches, patch_len)  # (b, c, n_patches, patch_len)
    x = x.permute(0, 2, 1, 3).contiguous()  # (b, n_patches, c, patch_len)
    return x.view(b, n_patches, c * patch_len)  # (b, n_patches, c*patch_len)


class TransformerStack(nn.Module):
    """A vanilla pre-norm transformer encoder stack, shared by the LeJEPA
    encoder and predictor at different widths/depths."""

    def __init__(self, d_model: int, n_heads: int, n_layers: int, ff_mult: int, dropout: float):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.net = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.net(x))


class LeJEPAEncoder(nn.Module):
    """The reusable part: patch embed -> transformer -> per-patch embeddings.

    This is what you keep after pretraining (drop the predictor).
    """

    def __init__(self, cfg: LeJEPAConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = nn.Linear(cfg.n_channels * cfg.patch_len, cfg.d_model)
        self.backbone = TransformerStack(
            cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.ff_mult, cfg.dropout
        )

    def embed_tokens(self, x: Tensor, sample_rate_hz: Tensor) -> Tensor:
        """(batch, n_channels, window_len) -> patch tokens (batch, n_patches, d_model)."""
        patches = patchify(x, self.cfg.patch_len)  # (b, n_patches, c*patch_len)
        tokens = self.patch_embed(patches)  # (b, n_patches, d_model)
        t = patch_times(self.cfg.window_len, self.cfg.patch_len, sample_rate_hz)
        return tokens + time_positional_encoding(t, self.cfg.d_model)

    def forward(self, x: Tensor, sample_rate_hz: Tensor) -> Tensor:
        """Full (unmasked) pass — used both for inference and as the 'target' branch."""
        tokens = self.embed_tokens(x, sample_rate_hz)
        return self.backbone(tokens)  # (b, n_patches, d_model)

    def forward_subset(self, x: Tensor, sample_rate_hz: Tensor, patch_idx: Tensor) -> Tensor:
        """Context pass — only the selected patches are embedded and attended over."""
        tokens = self.embed_tokens(x, sample_rate_hz)
        tokens = tokens[:, patch_idx, :]
        return self.backbone(tokens)
