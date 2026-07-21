import torch
import torch.nn.functional as F
from typing import Tuple


def _cubic_weight(t: torch.Tensor) -> torch.Tensor:
    at = t.abs()
    at2 = at * at
    at3 = at2 * at
    w1 = 1.5 * at3 - 2.5 * at2 + 1.0
    w2 = -0.5 * at3 + 2.5 * at2 - 4.0 * at + 2.0
    return torch.where(at <= 1.0, w1, torch.where(at <= 2.0, w2, torch.zeros_like(t)))


def _tricubic_warp_chunk(
    image: torch.Tensor,
    displacement_pix: torch.Tensor,
    d_start: int,
    d_end: int,
) -> torch.Tensor:
    B, C, D, H, W = image.shape
    device = image.device
    dtype = image.dtype
    chunk_d = d_end - d_start

    coords_d = torch.arange(d_start, d_end, device=device, dtype=dtype)
    coords_h = torch.arange(H, device=device, dtype=dtype)
    coords_w = torch.arange(W, device=device, dtype=dtype)
    grid_d, grid_h, grid_w = torch.meshgrid(
        coords_d, coords_h, coords_w, indexing='ij'
    )

    disp_d = displacement_pix[:, 0, d_start:d_end]
    disp_h = displacement_pix[:, 1, d_start:d_end]
    disp_w = displacement_pix[:, 2, d_start:d_end]

    sample_d = grid_d.unsqueeze(0) + disp_d
    sample_h = grid_h.unsqueeze(0) + disp_h
    sample_w = grid_w.unsqueeze(0) + disp_w

    floor_d = sample_d.detach().floor()
    floor_h = sample_h.detach().floor()
    floor_w = sample_w.detach().floor()

    frac_d = sample_d - floor_d
    frac_h = sample_h - floor_h
    frac_w = sample_w - floor_w

    floor_d_long = floor_d.long()
    floor_h_long = floor_h.long()
    floor_w_long = floor_w.long()

    image_flat = image.reshape(B, C, D * H * W)

    result = torch.zeros(B, C, chunk_d, H, W, device=device, dtype=dtype)

    for od in range(-1, 3):
        wd = _cubic_weight(frac_d - od)
        idx_d = (floor_d_long + od).clamp(0, D - 1)

        for oh in range(-1, 3):
            wdh = wd * _cubic_weight(frac_h - oh)
            idx_h = (floor_h_long + oh).clamp(0, H - 1)

            for ow in range(-1, 3):
                weight = wdh * _cubic_weight(frac_w - ow)
                idx_w = (floor_w_long + ow).clamp(0, W - 1)

                flat_idx = idx_d * (H * W) + idx_h * W + idx_w
                flat_idx_exp = flat_idx.reshape(B, 1, -1).expand(B, C, -1)
                sampled = torch.gather(image_flat, 2, flat_idx_exp)
                sampled = sampled.reshape(B, C, chunk_d, H, W)

                result = result + weight.unsqueeze(1) * sampled

    return result


def _tricubic_warp(
    image: torch.Tensor,
    displacement_pix: torch.Tensor,
    chunk_size: int = 16,
) -> torch.Tensor:
    D = image.shape[2]
    chunks = []

    for d_start in range(0, D, chunk_size):
        d_end = min(d_start + chunk_size, D)
        chunk = _tricubic_warp_chunk(image, displacement_pix, d_start, d_end)
        chunks.append(chunk)

    return torch.cat(chunks, dim=2)


def warp_image(
    image: torch.Tensor,
    displacement_pix: torch.Tensor,
    interp_type: int = 0,
) -> torch.Tensor:
    nd = image.dim() - 2

    if nd == 3 and interp_type == 1:
        return _tricubic_warp(image, displacement_pix, chunk_size=16)

    spatial_size = image.shape[2:]
    device = image.device
    dtype = image.dtype

    vectors = [
        torch.linspace(-1.0, 1.0, s, device=device, dtype=dtype)
        for s in spatial_size
    ]
    grids = torch.meshgrid(*vectors, indexing="ij")

    disp_norm = []
    for d in range(nd):
        s = spatial_size[d]
        scale = 2.0 / (s - 1) if s > 1 else 0.0
        disp_norm.append(displacement_pix[:, d] * scale)

    grid_components = []
    for gs_idx in range(nd):
        our_dim = nd - 1 - gs_idx
        identity = grids[our_dim].unsqueeze(0)
        sampling = identity + disp_norm[our_dim]
        grid_components.append(sampling)

    grid = torch.stack(grid_components, dim=-1)

    if nd == 2 and interp_type == 1:
        mode = "bicubic"
    elif nd == 2:
        mode = "bilinear"
    else:
        mode = "bilinear"

    warped = F.grid_sample(
        image, grid, mode=mode,
        padding_mode="border", align_corners=True,
    )

    return warped
