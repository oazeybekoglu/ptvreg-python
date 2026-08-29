import torch
from typing import Optional, Tuple


def ssd_loss(
    warped: torch.Tensor,
    fixed: torch.Tensor,
    pix_resolution: Tuple[float, ...],
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    diff_sq = (warped - fixed).pow(2)

    if mask is not None:
        diff_sq = diff_sq * mask
        n_eff = mask.sum().clamp(min=1.0)
        return 0.5 * diff_sq.sum() / n_eff

    return 0.5 * diff_sq.mean()


def sad_loss(
    warped: torch.Tensor,
    fixed: torch.Tensor,
    pix_resolution: Tuple[float, ...],
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    diff_abs = (warped - fixed).abs()

    if mask is not None:
        diff_abs = diff_abs * mask
        n_eff = mask.sum().clamp(min=1.0)
        return diff_abs.sum() / n_eff

    return diff_abs.mean()
