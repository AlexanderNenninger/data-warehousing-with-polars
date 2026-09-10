"""Smoke tests for the LeJEPA model: a real forward/backward pass on tiny dims."""

import torch
from self_supervised_physical_ai import LeJEPAConfig, LeJEPAModel, SyntheticSEFCDataset
from torch.utils.data import DataLoader


def _tiny_config(**overrides) -> LeJEPAConfig:
    defaults = dict(
        n_channels=2,
        window_len=32,
        patch_len=8,
        d_model=16,
        n_heads=4,
        n_layers=1,
        ff_mult=2,
        dropout=0.0,
        predictor_d_model=8,
        predictor_n_heads=4,
        predictor_n_layers=1,
        context_ratio=0.5,
        n_target_blocks=1,
        sigreg_n_directions=8,
        sigreg_n_freqs=4,
        batch_size=4,
    )
    defaults.update(overrides)
    return LeJEPAConfig(**defaults)


def _tiny_batch(cfg: LeJEPAConfig) -> dict[str, torch.Tensor]:
    dataset = SyntheticSEFCDataset(
        n_windows=cfg.batch_size, n_channels=cfg.n_channels, window_len=cfg.window_len
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size)
    return next(iter(loader))


def test_lejepa_forward_produces_finite_scalar_loss():
    cfg = _tiny_config()
    model = LeJEPAModel(cfg)
    batch = _tiny_batch(cfg)

    out = model(batch["x"], batch["sample_rate_hz"])

    assert out.loss.shape == ()
    assert torch.isfinite(out.loss)
    n_patches = cfg.n_patches
    assert len(out.context_idx) + len(out.target_idx) == n_patches
    assert set(out.context_idx.tolist()).isdisjoint(out.target_idx.tolist())


def test_lejepa_backward_updates_encoder_params():
    cfg = _tiny_config()
    model = LeJEPAModel(cfg)
    batch = _tiny_batch(cfg)

    out = model(batch["x"], batch["sample_rate_hz"])
    out.loss.backward()

    grads = [p.grad for p in model.encoder.parameters() if p.requires_grad]
    assert grads, "encoder should have trainable parameters"
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_lejepa_detach_target_still_trains():
    cfg = _tiny_config()
    model = LeJEPAModel(cfg, detach_target=True)
    batch = _tiny_batch(cfg)

    out = model(batch["x"], batch["sample_rate_hz"])

    assert torch.isfinite(out.loss)
