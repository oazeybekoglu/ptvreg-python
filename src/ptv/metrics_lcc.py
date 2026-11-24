import torch
import math
from typing import Tuple, Optional, Sequence

from .channel_weights import normalize_channel_weights


def _build_gaussian_fft(
    spatial_size: Tuple[int, ...],
    sigma_pix: Tuple[float, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    nd = len(spatial_size)


    gaussians_1d = []
    for d in range(nd):
        n = spatial_size[d]
        s = max(sigma_pix[d], 1e-8)
        coords = torch.arange(n, device=device, dtype=dtype)

        coords = torch.where(coords > n // 2, coords - n, coords)
        g = torch.exp(-0.5 * (coords / s) ** 2)
        gaussians_1d.append(g)


    if nd == 2:
        kernel = gaussians_1d[0].unsqueeze(1) * gaussians_1d[1].unsqueeze(0)
    elif nd == 3:
        kernel = (
            gaussians_1d[0].unsqueeze(1).unsqueeze(2)
            * gaussians_1d[1].unsqueeze(0).unsqueeze(2)
            * gaussians_1d[2].unsqueeze(0).unsqueeze(1)
        )
    else:
        raise ValueError(f"nd must be 2 or 3, got {nd}")


    kernel = kernel / kernel.sum()


    return torch.fft.rfftn(kernel)


def _gauss_smooth(
    x: torch.Tensor,
    g_fft: torch.Tensor,
    spatial_size: Tuple[int, ...],
    smooth_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    dims = list(range(2, 2 + len(spatial_size)))
    if smooth_dtype is not None and smooth_dtype != x.dtype:
        x_c = x.to(smooth_dtype)

        g_c = g_fft.to(torch.complex128 if smooth_dtype == torch.float64
                       else torch.complex64)
        x_fft = torch.fft.rfftn(x_c, dim=dims)
        out = torch.fft.irfftn(x_fft * g_c, s=spatial_size, dim=dims)


        return out

    x_fft = torch.fft.rfftn(x, dim=dims)
    return torch.fft.irfftn(x_fft * g_fft, s=spatial_size, dim=dims)


def lcc_loss(
    warped: torch.Tensor,
    fixed: torch.Tensor,
    pix_resolution: Tuple[float, ...],
    sigma_pix: Tuple[float, ...],
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-4,
    use_abs: bool = False,
    compute_dtype: Optional[torch.dtype] = None,
    sum_dtype: Optional[torch.dtype] = torch.float64,
    return_dtype: Optional[torch.dtype] = None,
    channel_weights: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """Negated local (Gaussian-window) correlation coefficient, summed over voxels.

    Multi-channel images (B, C, *spatial) are compared channel by channel
    (channel c of ``warped`` against channel c of ``fixed``) and the per-channel
    losses are combined with ``channel_weights`` (normalised to sum to 1;
    None = plain average over channels).

    sum_dtype : dtype of the voxel sum (float64 by default, so the reduction is
        exact even when compute_dtype is float32). None -> compute dtype.
    return_dtype : dtype of the returned scalar. None -> input dtype.
    """
    nd = warped.dim() - 2
    spatial_size = warped.shape[2:]
    device = warped.device
    dtype = warped.dtype
    n_channels = warped.shape[1]
    if fixed.shape[1] != n_channels:
        raise ValueError(
            f"channel mismatch: warped has {n_channels}, fixed has "
            f"{fixed.shape[1]} channel(s)")
    weights = normalize_channel_weights(channel_weights, n_channels)


    cdt = compute_dtype if compute_dtype is not None else dtype
    sdt = sum_dtype if sum_dtype is not None else cdt

    pix_vol = 1.0
    for p in pix_resolution:
        pix_vol *= p


    # Cast to the compute dtype BEFORE forming I*I, J*J, I*J and build the
    # Gaussian kernel in it too. var = E[I^2] - E[I]^2 cancels in flat
    # regions, so float32 products or kernel weights give ~1e-4 relative
    # error in the loss even when compute_dtype is float64.
    if cdt != dtype:
        warped = warped.to(cdt)
        fixed = fixed.to(cdt)

    g_fft = _build_gaussian_fft(spatial_size, sigma_pix[:nd], device, cdt)

    total_loss = torch.zeros((), device=device, dtype=sdt)

    for ch in range(n_channels):
        w_ch = weights[ch]
        if w_ch == 0.0:
            continue
        I = warped[:, ch:ch+1]
        J = fixed[:, ch:ch+1]


        gI  = _gauss_smooth(I, g_fft, spatial_size)
        gJ  = _gauss_smooth(J, g_fft, spatial_size)
        gII = _gauss_smooth(I * I, g_fft, spatial_size)
        gJJ = _gauss_smooth(J * J, g_fft, spatial_size)
        gIJ = _gauss_smooth(I * J, g_fft, spatial_size)


        var_I  = (gII - gI * gI).clamp(min=0.0)
        var_J  = (gJJ - gJ * gJ).clamp(min=0.0)
        cov_IJ = gIJ - gI * gJ


        denom = torch.sqrt(var_I + eps) * torch.sqrt(var_J + eps)
        cc = cov_IJ / denom

        if use_abs:
            cc = cc.abs()

        if mask is not None:
            cc = cc * mask.to(cdt)


        total_loss = total_loss - (w_ch * pix_vol) * cc.to(sdt).sum()


    return total_loss.to(return_dtype if return_dtype is not None else dtype)
