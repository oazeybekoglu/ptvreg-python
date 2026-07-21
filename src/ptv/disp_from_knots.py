import torch
import torch.nn.functional as F
from typing import Tuple

from .get_grid_size import get_grid_size


def disp_from_knots(
    knots: torch.Tensor,
    image_size: Tuple[int, ...],
    grid_spacing: Tuple[int, ...],
    spline_order: int = 1,
) -> torch.Tensor:
    if spline_order != 1:
        raise NotImplementedError(f"spline_order={spline_order} not yet ported")

    nd = len(image_size)
    ksz = knots.shape[2:]

    covered_size = tuple(
        (ksz[d] - 1) * grid_spacing[d] + 1 for d in range(nd)
    )

    if nd == 2:
        mode = "bilinear"
    elif nd == 3:
        mode = "trilinear"
    else:
        raise ValueError(f"nd must be 2 or 3, got {nd}")

    displacement = F.interpolate(
        knots, size=covered_size, mode=mode, align_corners=True,
    )

    if nd == 2:
        displacement = displacement[:, :, :image_size[0], :image_size[1]]
    elif nd == 3:
        displacement = displacement[:, :, :image_size[0], :image_size[1], :image_size[2]]

    return displacement


def init_knots(
    image_size: Tuple[int, ...],
    grid_spacing: Tuple[int, ...],
    nd: int,
    batch: int = 1,
    spline_order: int = 1,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    ksz = get_grid_size(image_size[:nd], grid_spacing[:nd], spline_order)
    knots = torch.zeros(
        (batch, nd, *ksz), device=device, dtype=dtype, requires_grad=True
    )
    return knots
