import torch
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PyramidLevel:
    image: torch.Tensor
    pix_resolution: Tuple[float, ...]
    image_size: Tuple[int, ...]


def _gaussian_blur_2d(image, sigma):
    if sigma < 0.1:
        return image


    ksz = int(2 * math.ceil(2 * sigma) + 1)
    ksz = max(ksz, 5)
    half = ksz // 2


    x = torch.arange(-half, half + 1, device=image.device, dtype=image.dtype)
    g1d = torch.exp(-0.5 * (x / sigma) ** 2)
    g1d = g1d / g1d.sum()


    kernel = g1d.unsqueeze(1) * g1d.unsqueeze(0)
    kernel = kernel.unsqueeze(0).unsqueeze(0)


    B, C, H, W = image.shape

    padded = F.pad(image, [half, half, half, half], mode='reflect')

    kernel_expanded = kernel.expand(C, -1, -1, -1)
    blurred = F.conv2d(padded, kernel_expanded, groups=C)
    return blurred


def _gaussian_blur_3d(image, sigma):
    if sigma < 0.1:
        return image

    ksz = int(2 * math.ceil(2 * sigma) + 1)
    ksz = max(ksz, 5)
    half = ksz // 2

    x = torch.arange(-half, half + 1, device=image.device, dtype=image.dtype)
    g1d = torch.exp(-0.5 * (x / sigma) ** 2)
    g1d = g1d / g1d.sum()


    kernel = (g1d.reshape(-1, 1, 1)
              * g1d.reshape(1, -1, 1)
              * g1d.reshape(1, 1, -1))
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    B, C, D, H, W = image.shape
    padded = F.pad(image, [half, half, half, half, half, half], mode='reflect')
    kernel_expanded = kernel.expand(C, -1, -1, -1, -1)
    blurred = F.conv3d(padded, kernel_expanded, groups=C)
    return blurred


def create_pyramid(
    image: torch.Tensor,
    nlvl: int,
    pix_resolution: Tuple[float, ...],
    k_down: float = 0.7,
) -> List[PyramidLevel]:
    nd = image.dim() - 2
    device = image.device
    dtype = image.dtype


    levels = [PyramidLevel(
        image=image,
        pix_resolution=tuple(pix_resolution[:nd]),
        image_size=tuple(image.shape[2:]),
    )]


    sigma = 0.4 * (0.5 / k_down)

    current = image
    current_pix_res = tuple(pix_resolution[:nd])

    for lvl in range(1, nlvl):

        if nd == 2:
            blurred = _gaussian_blur_2d(current, sigma)
        else:
            blurred = _gaussian_blur_3d(current, sigma)


        cur_spatial = current.shape[2:]
        new_size = tuple(math.ceil(k_down * s) for s in cur_spatial)


        mode = "bilinear" if nd == 2 else "trilinear"
        downscaled = F.interpolate(
            blurred, size=new_size, mode=mode, align_corners=True
        )


        new_pix_res = tuple(p / k_down for p in current_pix_res)

        levels.append(PyramidLevel(
            image=downscaled,
            pix_resolution=new_pix_res,
            image_size=new_size,
        ))

        current = downscaled
        current_pix_res = new_pix_res

    return levels


def compute_nlvl(
    image_size: Tuple[int, ...],
    grid_spacing: Tuple[int, ...],
    k_down: float = 0.7,
) -> int:
    nd = len(image_size)
    log_kd = abs(math.log(k_down))
    ratios = []
    for d in range(nd):
        gs = max(4, grid_spacing[d])
        ratios.append(int(math.floor(abs(math.log(image_size[d] / gs)) / log_kd)))
    return max(1, min(ratios))
