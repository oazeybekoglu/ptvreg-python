import torch
import math
from typing import Tuple, Optional


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
) -> torch.Tensor:
    x_fft = torch.fft.rfftn(x, dim=list(range(2, 2 + len(spatial_size))))
    return torch.fft.irfftn(x_fft * g_fft, s=spatial_size,
                            dim=list(range(2, 2 + len(spatial_size))))


def lcc_loss(
    warped: torch.Tensor,
    fixed: torch.Tensor,
    pix_resolution: Tuple[float, ...],
    sigma_pix: Tuple[float, ...],
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-4,
    use_abs: bool = False,
) -> torch.Tensor:
    nd = warped.dim() - 2
    spatial_size = warped.shape[2:]
    device = warped.device
    dtype = warped.dtype
    n_channels = warped.shape[1]

    pix_vol = 1.0
    for p in pix_resolution:
        pix_vol *= p

    g_fft = _build_gaussian_fft(spatial_size, sigma_pix[:nd], device, dtype)

    total_loss = torch.tensor(0.0, device=device, dtype=dtype)

    for ch in range(n_channels):
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
            cc = cc * mask

        total_loss = total_loss - pix_vol * cc.sum()

    total_loss = total_loss / n_channels

    return total_loss
