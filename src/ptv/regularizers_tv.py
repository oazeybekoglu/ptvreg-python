import torch
from typing import Tuple


def _my_gradient(x: torch.Tensor, h: float, dim: int) -> torch.Tensor:
    n = x.shape[dim]
    ndim = x.dim()

    if n <= 1:
        return torch.zeros_like(x)

    def _slice(d, s):
        slices = [slice(None)] * ndim
        slices[d] = s
        return tuple(slices)

    fwd = (x[_slice(dim, slice(1, None))] - x[_slice(dim, slice(None, -1))]) / h

    result = torch.empty_like(x)
    result[_slice(dim, slice(None, -1))] = fwd
    result[_slice(dim, -1)] = fwd[_slice(dim, -1)]

    return result


def isotropic_tv_regularization(
    knots: torch.Tensor,
    spacing: Tuple[float, ...],
    csqrt: float = 5e-3,
) -> torch.Tensor:
    nd = knots.dim() - 2
    batch = knots.shape[0]
    n_components = knots.shape[1]
    ksz = knots.shape[2:]

    vol = 1.0
    for s in spacing[:nd]:
        vol *= s

    sum_sq = torch.zeros(
        (batch, 1) + ksz, device=knots.device, dtype=knots.dtype
    )

    for comp in range(n_components):
        for j in range(nd):
            g = _my_gradient(
                knots[:, comp : comp + 1],
                spacing[j],
                dim=j + 2,
            )
            sum_sq = sum_sq + g ** 2


    tv_vals = torch.sqrt(sum_sq + csqrt)

    return vol * tv_vals.sum() / batch
