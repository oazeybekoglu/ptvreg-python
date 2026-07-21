import torch
from typing import Tuple


def params_phys_to_pix(
    knots: torch.Tensor,
    pix_resolution: Tuple[float, ...],
    spline_order: int = 1,
) -> torch.Tensor:
    nd = knots.shape[1]

    if spline_order in (1, 3):
        pix_res = torch.tensor(
            pix_resolution[:nd], device=knots.device, dtype=knots.dtype
        )
        shape = [1, nd] + [1] * (knots.dim() - 2)
        pix_res = pix_res.view(shape)
        knots_pix = knots / pix_res
    else:
        raise ValueError(f"Unsupported spline_order={spline_order}")

    return knots_pix
