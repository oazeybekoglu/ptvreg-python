from typing import Tuple, Optional

import torch
import torch.nn.functional as F


_IDENTITY_CACHE = {}


def _identity_grid_norm(
    spatial_size: Tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, ...]:
    key = (spatial_size, str(device), dtype)
    cached = _IDENTITY_CACHE.get(key)
    if cached is not None:
        return cached

    vectors = tuple(
        torch.linspace(-1.0, 1.0, s, device=device, dtype=dtype)
        for s in spatial_size
    )
    _IDENTITY_CACHE[key] = vectors
    return vectors


def compose_displacements(
    disp_outer_pix: torch.Tensor,
    disp_inner_pix: torch.Tensor,
) -> torch.Tensor:
    if disp_outer_pix.shape != disp_inner_pix.shape:
        raise ValueError(
            f"shape mismatch: outer={tuple(disp_outer_pix.shape)} "
            f"inner={tuple(disp_inner_pix.shape)}"
        )

    nd = disp_outer_pix.dim() - 2
    if nd not in (2, 3):
        raise ValueError(f"expected nd in (2, 3), got nd={nd}")

    spatial_size = tuple(disp_outer_pix.shape[2:])
    device = disp_outer_pix.device
    dtype = disp_outer_pix.dtype


    identity_axes = _identity_grid_norm(spatial_size, device, dtype)


    disp_inner_norm = []
    for d in range(nd):
        s = spatial_size[d]
        scale = 2.0 / (s - 1) if s > 1 else 0.0
        disp_inner_norm.append(disp_inner_pix[:, d] * scale)


    grid_components = []
    for gs_idx in range(nd):
        our_dim = nd - 1 - gs_idx

        identity_axis = identity_axes[our_dim]

        view_shape = [1] * nd
        view_shape[our_dim] = spatial_size[our_dim]
        identity_full = identity_axis.view(view_shape).expand(*spatial_size)

        sampling = identity_full.unsqueeze(0) + disp_inner_norm[our_dim]
        grid_components.append(sampling)


    grid = torch.stack(grid_components, dim=-1)


    disp_outer_resampled = F.grid_sample(
        disp_outer_pix,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )

    return disp_inner_pix + disp_outer_resampled


def composition_validity_mask(
    disp_inner_pix: torch.Tensor,
    spatial_size: Tuple[int, ...],
    margin: float = 0.0,
) -> torch.Tensor:
    nd = disp_inner_pix.dim() - 2
    if nd not in (2, 3):
        raise ValueError(f"expected nd in (2, 3), got nd={nd}")
    if len(spatial_size) != nd:
        raise ValueError(
            f"spatial_size length {len(spatial_size)} != nd={nd}"
        )

    device = disp_inner_pix.device
    dtype = disp_inner_pix.dtype
    B = disp_inner_pix.shape[0]

    valid = torch.ones(
        (B, 1) + tuple(spatial_size), device=device, dtype=dtype
    )

    for d in range(nd):
        s = spatial_size[d]
        coords = torch.arange(s, device=device, dtype=dtype)
        view_shape = [1, 1] + [1] * nd
        view_shape[d + 2] = s
        coords_b = coords.view(view_shape)
        target_pos = coords_b + disp_inner_pix[:, d:d + 1]
        in_bounds_d = ((target_pos >= margin)
                       & (target_pos <= s - 1 - margin)).to(dtype)
        valid = valid * in_bounds_d

    return valid


def inverse_consistency_loss(
    disp_FM_pix: torch.Tensor,
    disp_MF_pix: torch.Tensor,
    pix_resolution: Tuple[float, ...],
    mask: Optional[torch.Tensor] = None,
    reduction: str = "sum_voxel_volume",
    mask_out_of_bounds: bool = True,
    oob_margin: float = 0.0,
) -> torch.Tensor:
    if disp_FM_pix.shape != disp_MF_pix.shape:
        raise ValueError(
            f"shape mismatch: FM={tuple(disp_FM_pix.shape)} "
            f"MF={tuple(disp_MF_pix.shape)}"
        )
    nd = disp_FM_pix.dim() - 2
    if len(pix_resolution) < nd:
        raise ValueError(
            f"pix_resolution has length {len(pix_resolution)}, expected >= {nd}"
        )


    res_fwd_pix = compose_displacements(disp_MF_pix, disp_FM_pix)
    res_bwd_pix = compose_displacements(disp_FM_pix, disp_MF_pix)


    pr = torch.tensor(
        pix_resolution[:nd],
        device=disp_FM_pix.device,
        dtype=disp_FM_pix.dtype,
    ).view(1, nd, *([1] * nd))

    res_fwd_mm = res_fwd_pix * pr
    res_bwd_mm = res_bwd_pix * pr


    sq_fwd = (res_fwd_mm ** 2).sum(dim=1, keepdim=True)
    sq_bwd = (res_bwd_mm ** 2).sum(dim=1, keepdim=True)


    spatial_size = tuple(disp_FM_pix.shape[2:])
    if mask_out_of_bounds:
        valid_fwd = composition_validity_mask(
            disp_FM_pix, spatial_size, margin=oob_margin
        ).detach()
        valid_bwd = composition_validity_mask(
            disp_MF_pix, spatial_size, margin=oob_margin
        ).detach()
        sq_fwd = sq_fwd * valid_fwd
        sq_bwd = sq_bwd * valid_bwd

    if mask is not None:
        sq_fwd = sq_fwd * mask
        sq_bwd = sq_bwd * mask

    if reduction == "sum_voxel_volume":


        v = 1.0
        for p in pix_resolution[:nd]:
            v *= p
        return 0.5 * v * (sq_fwd.sum() + sq_bwd.sum())

    if reduction == "sum":
        return sq_fwd.sum() + sq_bwd.sum()

    if reduction == "mean":


        if mask_out_of_bounds and mask is not None:
            denom_fwd = (valid_fwd * mask).sum().clamp(min=1.0)
            denom_bwd = (valid_bwd * mask).sum().clamp(min=1.0)
            return (sq_fwd.sum() / denom_fwd + sq_bwd.sum() / denom_bwd) * 0.5
        if mask_out_of_bounds:
            denom_fwd = valid_fwd.sum().clamp(min=1.0)
            denom_bwd = valid_bwd.sum().clamp(min=1.0)
            return (sq_fwd.sum() / denom_fwd + sq_bwd.sum() / denom_bwd) * 0.5
        if mask is not None:
            denom = mask.sum().clamp(min=1.0)
            return (sq_fwd.sum() + sq_bwd.sum()) / (2.0 * denom)
        return 0.5 * (sq_fwd.mean() + sq_bwd.mean())

    raise ValueError(
        f"unknown reduction='{reduction}'. "
        "Use 'sum_voxel_volume', 'sum', or 'mean'."
    )
