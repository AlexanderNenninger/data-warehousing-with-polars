"""Top-level LeJEPA model: predictor + SIGReg-regularized encoder training.

DELIBERATE DESIGN CHOICES TO REVISIT (not settled decisions):

1. Masking style: I-JEPA-style patch masking + predictor, not LeJEPA's
   original view-augmentation setup (global/local crops). Patch masking fits
   "discover operating states in a continuous stream" better than
   image-style crops, but it's a hybrid of two papers' ideas, not a
   reproduction of either.
2. No stop-gradient on the target branch (`detach_target=False` by default).
   This follows LeJEPA's claim that SIGReg alone prevents collapse — worth
   A/B testing against `detach_target=True` on real data before trusting it,
   especially since SIGReg here is simplified (see losses.py).
3. Shared mask across the batch (views.sample_shared_mask) rather than
   per-sample masks — implementation simplicity for the POC. Per-sample
   masking is the first thing to fix before any real training run.
4. SIGReg is applied per-patch-embedding (all patches across the batch,
   flattened), not on a pooled/global embedding — plenty of samples for the
   characteristic-function test even at small batch size, but regularizes
   the *local* patch representation distribution rather than a per-window
   summary.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from self_supervised_physical_ai.config import LeJEPAConfig
from self_supervised_physical_ai.encoders import (
    LeJEPAEncoder,
    TransformerStack,
    patch_times,
    time_positional_encoding,
)
from self_supervised_physical_ai.losses import prediction_loss, sigreg_loss
from self_supervised_physical_ai.views import sample_shared_mask


class Predictor(nn.Module):
    """Small transformer: context embeddings + mask tokens (at target
    positions) -> predicted target embeddings."""

    def __init__(self, cfg: LeJEPAConfig):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.d_model, cfg.predictor_d_model)
        self.out_proj = nn.Linear(cfg.predictor_d_model, cfg.d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.predictor_d_model))
        nn.init.normal_(self.mask_token, std=0.02)
        self.backbone = TransformerStack(
            cfg.predictor_d_model,
            cfg.predictor_n_heads,
            cfg.predictor_n_layers,
            cfg.ff_mult,
            cfg.dropout,
        )

    def forward(self, context_emb: Tensor, context_pos: Tensor, target_pos: Tensor) -> Tensor:
        b, n_target, _ = target_pos.shape
        ctx = self.in_proj(context_emb) + self.in_proj(context_pos)
        mask_tok = self.mask_token.expand(b, n_target, -1) + self.in_proj(target_pos)
        seq = torch.cat([ctx, mask_tok], dim=1)
        out = self.backbone(seq)
        pred_target = out[:, ctx.shape[1] :, :]
        return self.out_proj(pred_target)  # (b, n_target, d_model)


@dataclass
class LeJEPAOutput:
    loss: Tensor
    pred_loss: Tensor
    sigreg_loss: Tensor
    context_idx: Tensor
    target_idx: Tensor


class LeJEPAModel(nn.Module):
    def __init__(self, cfg: LeJEPAConfig, detach_target: bool = False):
        super().__init__()
        self.cfg = cfg
        self.encoder = LeJEPAEncoder(cfg)
        self.predictor = Predictor(cfg)
        self.detach_target = detach_target

    def forward(self, x: Tensor, sample_rate_hz: Tensor) -> LeJEPAOutput:
        cfg = self.cfg
        device = x.device

        context_idx, target_idx = sample_shared_mask(
            cfg.n_patches, cfg.context_ratio, cfg.n_target_blocks, device
        )

        # Target branch: full (unmasked) encoder pass, same weights as context branch.
        full_emb = self.encoder(x, sample_rate_hz)  # (b, n_patches, d_model)
        target_emb = full_emb[:, target_idx, :]
        if self.detach_target:
            target_emb = target_emb.detach()

        # Context branch: only context patches are embedded/attended.
        context_emb = self.encoder.forward_subset(x, sample_rate_hz, context_idx)

        # Positional codes for the predictor (recomputed directly, not pulled from tokens,
        # so the predictor gets clean position info independent of content).
        t_all = patch_times(cfg.window_len, cfg.patch_len, sample_rate_hz)  # (b, n_patches)
        pos_all = time_positional_encoding(t_all, cfg.d_model)  # (b, n_patches, d_model)
        context_pos = pos_all[:, context_idx, :]
        target_pos = pos_all[:, target_idx, :]

        pred_target = self.predictor(context_emb, context_pos, target_pos)

        pred_loss = prediction_loss(pred_target, target_emb)

        # SIGReg regularizes the marginal distribution of patch embeddings toward
        # isotropic Gaussian; apply it over all patches (context+target) in the
        # batch — plenty of samples even with a small batch size.
        flat_emb = full_emb.reshape(-1, cfg.d_model)
        sig_loss = sigreg_loss(flat_emb, cfg.sigreg_n_directions, cfg.sigreg_n_freqs)

        total = pred_loss + cfg.sigreg_lambda * sig_loss
        return LeJEPAOutput(total, pred_loss, sig_loss, context_idx, target_idx)
