import torch
import torch.nn.functional as F
from typing import Tuple

from .get_grid_size import get_grid_size
from .disp_from_knots import disp_from_knots


def _resize_like_matlab(tensor, new_size, nd):
    mode = "bilinear" if nd == 2 else "trilinear"
    return F.interpolate(
        tensor, size=new_size, mode=mode, align_corners=True
    )


def upsample_knots(
    knots: torch.Tensor,
    image_size_prev: Tuple[int, ...],
    grid_spacing_prev: Tuple[int, ...],
    image_size_new: Tuple[int, ...],
    grid_spacing_new: Tuple[int, ...],
    spline_order: int = 1,
    method: str = "variational",
    refine_iters: int = 50,
) -> torch.Tensor:
    if spline_order != 1:
        raise NotImplementedError(f"spline_order={spline_order}")

    nd = knots.dim() - 2

    ksz_new = get_grid_size(
        image_size_new[:nd], grid_spacing_new[:nd], spline_order
    )


    knots_init = _resize_like_matlab(knots, ksz_new, nd)

    if method == "direct":
        return knots_init


    with torch.no_grad():
        disp_old = disp_from_knots(
            knots, image_size_prev[:nd], grid_spacing_prev[:nd], spline_order
        )


        target_disp = _resize_like_matlab(disp_old, image_size_new[:nd], nd)


    knots_opt = knots_init.detach().clone().requires_grad_(True)

    optimizer = torch.optim.LBFGS(
        [knots_opt],
        max_iter=1,
        history_size=35,
        line_search_fn="strong_wolfe",
    )

    for _ in range(refine_iters):
        def closure():
            optimizer.zero_grad()
            disp_pred = disp_from_knots(
                knots_opt, image_size_new[:nd], grid_spacing_new[:nd],
                spline_order,
            )
            loss = 0.5 * ((disp_pred - target_disp) ** 2).sum()
            loss.backward()
            return loss
        optimizer.step(closure)


    with torch.no_grad():
        disp_pred = disp_from_knots(
            knots_opt, image_size_new[:nd], grid_spacing_new[:nd], spline_order
        )
        mismatch = ((disp_pred - target_disp) ** 2).mean().sqrt().item()
        print(f"    upsample_knots: refinement RMSE = {mismatch:.6f}")
    
    return knots_opt.detach()
