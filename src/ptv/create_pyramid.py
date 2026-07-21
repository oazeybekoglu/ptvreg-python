import math
from typing import Tuple


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
