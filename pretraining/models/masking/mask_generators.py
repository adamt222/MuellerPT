from __future__ import annotations

import torch


def apply_channel_mask(
    x: torch.Tensor,
    channel_mask: torch.Tensor,
    mask_token: torch.Tensor,
) -> torch.Tensor:
    """Zero the selected Mueller channels using the historical corruption path."""
    if channel_mask.shape != x.shape[:2]:
        raise ValueError(
            f"Expected channel mask shape {tuple(x.shape[:2])}, "
            f"got {tuple(channel_mask.shape)}."
        )
    channel_mask = channel_mask.to(device=x.device, dtype=torch.bool)
    expanded = channel_mask[:, :, None, None].expand_as(x)

    # The parameter remains in the checkpoint for compatibility, but the
    # published run used no spatial masking and therefore did not use it.
    del mask_token
    x_corrupt = x.clone()
    return torch.where(expanded, torch.zeros_like(x_corrupt), x_corrupt)
