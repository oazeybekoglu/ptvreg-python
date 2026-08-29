import math
from typing import Tuple


def get_grid_size(
    image_size: Tuple[int, ...],
    grid_spacing: Tuple[int, ...],
    spline_order: int = 1,
) -> Tuple[int, ...]:

    nd = min(len(image_size), len(grid_spacing))
    imsz = image_size[:nd]
    gs = grid_spacing[:nd]

    if spline_order == 1:

        grid_sz = tuple(math.ceil(imsz[d] / gs[d]) + 1 for d in range(nd))
    elif spline_order == 3:

        grid_sz = tuple(math.ceil(imsz[d] / gs[d]) + 3 for d in range(nd))
    else:
        raise ValueError(f"Unsupported spline_order={spline_order}. Use 1 or 3.")

    return grid_sz
