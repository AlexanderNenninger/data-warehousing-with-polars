"""Losses for LeJEPA training: the SIGReg collapse-prevention regularizer and
the context->target prediction loss.

`sigreg_loss` is a compact reimplementation of SIGReg (Balestriero & LeCun,
"LeJEPA: Provable and Scalable Self-Supervised Learning Without the
Heuristics", arXiv:2511.08544), not a copy of the reference implementation at
github.com/rbalestr-lab/lejepa — the quadrature grid over t and exact
weighting differ. Treat this version as "architecturally faithful,
numerically approximate"; swap in the reference kernel if you need
paper-exact numbers.
"""

import torch
from torch import Tensor, nn


def sigreg_loss(
    z: Tensor,
    n_directions: int = 64,
    n_freqs: int = 8,
    t_max: float = 3.0,
) -> Tensor:
    """Sketched isotropic-Gaussian regularization.

    Args:
        z: (batch, d_model) embeddings for one view (post-encoder, pre-predictor
            or the target embeddings — SIGReg is applied to the marginal
            embedding distribution, not to a pair).
        n_directions: number of random 1D projections (sketching).
        n_freqs: quadrature points for the characteristic-function test.
        t_max: largest quadrature point.

    Returns:
        Scalar loss, ~0 when z's distribution matches an isotropic standard
        Gaussian, positive otherwise. Differentiable in z.
    """
    batch, d = z.shape
    device = z.device

    # Random unit directions (Cramer-Wold: matching all 1D projections
    # implies matching the full joint distribution).
    w = torch.randn(d, n_directions, device=device)
    w = w / w.norm(dim=0, keepdim=True).clamp_min(1e-8)

    s = z @ w  # (batch, n_directions)

    # Quadrature grid over t, symmetric around 0 (empirical char. fn. is
    # conjugate-symmetric for real-valued s, so t>0 suffices).
    t = torch.linspace(0.2, t_max, n_freqs, device=device)  # (n_freqs,)

    # Empirical characteristic function: phi_hat(t) = (1/N) sum_n exp(i t s_n)
    ts = s.unsqueeze(-1) * t.view(1, 1, -1)  # (batch, n_directions, n_freqs)
    cos_term = torch.cos(ts).mean(dim=0)  # (n_directions, n_freqs)
    sin_term = torch.sin(ts).mean(dim=0)  # (n_directions, n_freqs)

    # Standard normal characteristic function: phi(t) = exp(-t^2 / 2), purely real.
    target = torch.exp(-0.5 * t**2)  # (n_freqs,)

    sq_err = (cos_term - target.view(1, -1)) ** 2 + sin_term**2  # (n_directions, n_freqs)
    return sq_err.mean()


def prediction_loss(pred_target: Tensor, target_emb: Tensor) -> Tensor:
    """Smooth-L1 loss between predicted and actual target patch embeddings."""
    return nn.functional.smooth_l1_loss(pred_target, target_emb)
