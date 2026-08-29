import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def _bspline3(x: torch.Tensor) -> torch.Tensor:
    ax = x.abs()
    ax2 = ax * ax
    ax3 = ax2 * ax
    inner = (2.0 / 3.0) - ax2 + 0.5 * ax3
    outer = ((2.0 - ax) ** 3) / 6.0
    val = torch.where(ax < 1.0, inner, outer)
    return torch.where(ax < 2.0, val, torch.zeros_like(val))


def mi_loss(
    warped: torch.Tensor,
    fixed: torch.Tensor,
    pix_resolution: Tuple[float, ...],
    n_bins: int = 32,
    intensity_range: Tuple[float, float] = (0.0, 1.0),
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-10,
    min_voxels_per_bin: float = 50.0,
    verbose: bool = False,
) -> torch.Tensor:
    if warped.shape != fixed.shape:
        raise ValueError(
            f"shape mismatch: warped {tuple(warped.shape)} vs "
            f"fixed {tuple(fixed.shape)}"
        )
    if n_bins < 4:
        raise ValueError(f"n_bins must be >= 4 (got {n_bins})")

    device = warped.device
    dtype = warped.dtype


    f_flat = fixed.reshape(-1)
    w_flat = warped.reshape(-1)


    if mask is not None:
        m_flat = mask.reshape(-1)
        keep = m_flat > 0
        f_flat = f_flat[keep]
        w_flat = w_flat[keep]

    n_active = f_flat.numel()
    if n_active == 0:

        return (warped * 0.0).sum()


    if min_voxels_per_bin and min_voxels_per_bin > 0:
        import math as _math
        cap = int(_math.floor(_math.sqrt(n_active / float(min_voxels_per_bin))))
        eff_bins = max(6, min(int(n_bins), cap))
    else:
        eff_bins = int(n_bins)

    if verbose and eff_bins < n_bins:
        print(f"    MI: {n_active} voxels -> reduced bins {n_bins} -> {eff_bins} "
              f"(min_voxels_per_bin={min_voxels_per_bin})")


    n_bins = eff_bins


    lo, hi = float(intensity_range[0]), float(intensity_range[1])
    span = max(hi - lo, 1e-12)
    n_data_bins = n_bins - 4
    if n_data_bins < 2:
        raise ValueError(f"n_bins={n_bins} too small (need >= 6 with padding).")

    b_F = (f_flat - lo) / span * (n_data_bins - 1) + 2.0
    b_M = (w_flat - lo) / span * (n_data_bins - 1) + 2.0


    b_F = b_F.clamp(2.0, n_bins - 3.0)
    b_M = b_M.clamp(2.0, n_bins - 3.0)


    i_F = torch.floor(b_F).long()
    i_M = torch.floor(b_M).long()
    df  = b_F - i_F.to(dtype)
    dm  = b_M - i_M.to(dtype)


    offsets = torch.tensor([-1, 0, 1, 2], device=device)
    offsets_f = offsets.to(dtype)


    w_F_all = _bspline3(df.unsqueeze(-1) - offsets_f.unsqueeze(0))
    w_M_all = _bspline3(dm.unsqueeze(-1) - offsets_f.unsqueeze(0))


    joint_flat = torch.zeros(n_bins * n_bins, dtype=dtype, device=device)


    for k in range(4):
        for l in range(4):

            bin_F = i_F + offsets[k]
            bin_M = i_M + offsets[l]
            flat_idx = bin_F * n_bins + bin_M

            w_kl = w_F_all[:, k] * w_M_all[:, l]

            joint_flat = joint_flat.scatter_add(0, flat_idx, w_kl)


    joint = joint_flat.view(n_bins, n_bins) / float(n_active)


    p_F = joint.sum(dim=1)
    p_M = joint.sum(dim=0)


    H_F  = -(p_F * torch.log(p_F + eps)).sum()
    H_M  = -(p_M * torch.log(p_M + eps)).sum()
    H_FM = -(joint * torch.log(joint + eps)).sum()
    mi   = H_F + H_M - H_FM


    pix_vol = 1.0
    for p in pix_resolution:
        pix_vol *= float(p)

    return -mi * pix_vol * n_active
