"""Context/target masking: splits a window's patches into the two views LeJEPA trains on."""

import torch


def sample_shared_mask(
    n_patches: int,
    context_ratio: float,
    n_target_blocks: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Samples a context/target patch split, shared across the batch.

    Note: a shared mask means every window in a batch reveals/hides the same
    time region — the simplest thing to fix before any real training run is
    per-sample masking instead.

    Returns:
        (context_idx, target_idx) as 1D LongTensors.
    """
    n_target = max(1, int(round(n_patches * (1 - context_ratio))))
    block_len = max(1, n_target // n_target_blocks)
    target_idx: set[int] = set()
    tries = 0
    while len(target_idx) < n_target and tries < 100:
        start = int(torch.randint(0, n_patches, (1,)).item())
        block = range(start, min(start + block_len, n_patches))
        target_idx.update(block)
        tries += 1
    sorted_target = sorted(target_idx)[:n_target]
    context_idx = [i for i in range(n_patches) if i not in sorted_target]
    return (
        torch.tensor(context_idx, dtype=torch.long, device=device),
        torch.tensor(sorted_target, dtype=torch.long, device=device),
    )
