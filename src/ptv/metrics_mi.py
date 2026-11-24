import math
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Sequence

from .channel_weights import normalize_channel_weights


def _bspline3(x: torch.Tensor) -> torch.Tensor:
    ax = x.abs()
    ax2 = ax * ax
    ax3 = ax2 * ax
    inner = (2.0 / 3.0) - ax2 + 0.5 * ax3
    outer = ((2.0 - ax) ** 3) / 6.0
    val = torch.where(ax < 1.0, inner, outer)
    return torch.where(ax < 2.0, val, torch.zeros_like(val))


def _mi_single_channel(
    warped: torch.Tensor,
    fixed: torch.Tensor,
    pix_vol: float,
    n_bins: int,
    intensity_range: Tuple[float, float],
    mask: Optional[torch.Tensor],
    eps: float,
    min_voxels_per_bin: float,
    verbose: bool,
    cdt: torch.dtype,
    sdt: torch.dtype,
) -> torch.Tensor:
    """Negated Mattes MI of one channel pair, in the sum dtype ``sdt``."""
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

        return (warped * 0.0).sum().to(sdt)

    if cdt != dtype:
        f_flat = f_flat.to(cdt)
        w_flat = w_flat.to(cdt)


    if min_voxels_per_bin and min_voxels_per_bin > 0:
        cap = int(math.floor(math.sqrt(n_active / float(min_voxels_per_bin))))
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
    df  = b_F - i_F.to(cdt)
    dm  = b_M - i_M.to(cdt)


    offsets = torch.tensor([-1, 0, 1, 2], device=device)
    offsets_f = offsets.to(cdt)


    w_F_all = _bspline3(df.unsqueeze(-1) - offsets_f.unsqueeze(0))
    w_M_all = _bspline3(dm.unsqueeze(-1) - offsets_f.unsqueeze(0))


    # Joint histogram accumulated in sdt (float64 by default): millions of
    # voxels summed into <= n_bins^2 cells in float32 bias the MI value and
    # its gradient at full resolution.
    joint_flat = torch.zeros(n_bins * n_bins, dtype=sdt, device=device)


    for k in range(4):
        for l in range(4):

            bin_F = i_F + offsets[k]
            bin_M = i_M + offsets[l]
            flat_idx = bin_F * n_bins + bin_M

            w_kl = w_F_all[:, k] * w_M_all[:, l]

            joint_flat = joint_flat.scatter_add(0, flat_idx, w_kl.to(sdt))


    joint = joint_flat.view(n_bins, n_bins) / float(n_active)


    p_F = joint.sum(dim=1)
    p_M = joint.sum(dim=0)


    H_F  = -(p_F * torch.log(p_F + eps)).sum()
    H_M  = -(p_M * torch.log(p_M + eps)).sum()
    H_FM = -(joint * torch.log(joint + eps)).sum()
    mi   = H_F + H_M - H_FM

    return -mi * pix_vol * n_active


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
    compute_dtype: Optional[torch.dtype] = None,
    sum_dtype: Optional[torch.dtype] = torch.float64,
    return_dtype: Optional[torch.dtype] = None,
    channel_weights: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """Negated Mattes mutual information (B-spline Parzen windows).

    Multi-channel images (B, C, *spatial) are compared channel by channel:
    one joint histogram per channel pair (channel c of ``warped`` against
    channel c of ``fixed``), and the per-channel losses are combined with
    ``channel_weights`` (normalised to sum to 1; None = plain average).
    Every channel is binned over the same ``intensity_range``, so scale each
    channel to that range first. One channel gives the usual single MI.

    compute_dtype : dtype for the bin positions and Parzen weights.
        None -> input dtype.
    sum_dtype : dtype of the joint histogram and entropies (float64 by
        default). None -> compute dtype.
    return_dtype : dtype of the returned scalar. None -> input dtype.
    """
    if warped.shape != fixed.shape:
        raise ValueError(
            f"shape mismatch: warped {tuple(warped.shape)} vs "
            f"fixed {tuple(fixed.shape)}"
        )
    if n_bins < 4:
        raise ValueError(f"n_bins must be >= 4 (got {n_bins})")

    dtype = warped.dtype
    cdt = compute_dtype if compute_dtype is not None else dtype
    sdt = sum_dtype if sum_dtype is not None else cdt
    rdt = return_dtype if return_dtype is not None else dtype

    pix_vol = 1.0
    for p in pix_resolution:
        pix_vol *= float(p)

    n_channels = warped.shape[1]
    weights = normalize_channel_weights(channel_weights, n_channels)

    total = torch.zeros((), device=warped.device, dtype=sdt)
    first = True
    for ch in range(n_channels):
        w_ch = weights[ch]
        if w_ch == 0.0:
            continue
        mi_ch = _mi_single_channel(
            warped[:, ch:ch+1], fixed[:, ch:ch+1], pix_vol,
            n_bins, intensity_range, mask, eps, min_voxels_per_bin,
            verbose and first, cdt, sdt)
        total = total + w_ch * mi_ch
        first = False

    return total.to(rdt)
