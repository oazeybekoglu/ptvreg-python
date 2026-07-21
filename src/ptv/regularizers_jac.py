import torch
from typing import Tuple


def _forward_diff_same_size(x: torch.Tensor, h: float, dim: int) -> torch.Tensor:
    n = x.shape[dim]
    ndim = x.dim()

    if n <= 1:
        return torch.zeros_like(x)

    fwd = (x.narrow(dim, 1, n - 1) - x.narrow(dim, 0, n - 1)) / h

    result = torch.empty_like(x)

    slices_out = [slice(None)] * ndim
    slices_out[dim] = slice(None, -1)
    result[tuple(slices_out)] = fwd

    slices_last_out = [slice(None)] * ndim
    slices_last_out[dim] = -1
    slices_last_fwd = [slice(None)] * ndim
    slices_last_fwd[dim] = -1
    result[tuple(slices_last_out)] = fwd[tuple(slices_last_fwd)]

    return result


def jac_reg_regularization(
    knots: torch.Tensor,
    spacing: Tuple[float, ...],
    lower_bound: float = 0.2,
    upper_bound: float = 4.0,
) -> torch.Tensor:
    nd = knots.dim() - 2
    batch = knots.shape[0]

    vol = 1.0
    for s in spacing[:nd]:
        vol *= s

    grads = {}
    for i in range(nd):
        for j in range(nd):
            grads[(i, j)] = _forward_diff_same_size(
                knots[:, i:i+1], spacing[j], dim=j + 2
            )

    if nd == 2:
        j00 = 1.0 + grads[(0, 0)]
        j01 = grads[(0, 1)]
        j10 = grads[(1, 0)]
        j11 = 1.0 + grads[(1, 1)]
        det_J = j00 * j11 - j01 * j10
    elif nd == 3:
        j00 = 1.0 + grads[(0, 0)]
        j01 = grads[(0, 1)]
        j02 = grads[(0, 2)]
        j10 = grads[(1, 0)]
        j11 = 1.0 + grads[(1, 1)]
        j12 = grads[(1, 2)]
        j20 = grads[(2, 0)]
        j21 = grads[(2, 1)]
        j22 = 1.0 + grads[(2, 2)]
        det_J = (j00 * (j11 * j22 - j12 * j21)
               - j01 * (j10 * j22 - j12 * j20)
               + j02 * (j10 * j21 - j11 * j20))
    else:
        raise ValueError(f"nd must be 2 or 3, got {nd}")

    penalty_low = torch.clamp(lower_bound - det_J, min=0.0) ** 2
    penalty_high = torch.clamp(det_J - upper_bound, min=0.0) ** 2
    total = (penalty_low + penalty_high).sum()

    return 0.5 * vol * total / batch
