import math
from typing import List, Optional, Sequence, Union

import torch


def normalize_channel_weights(
    channel_weights: Optional[Union[Sequence[float], torch.Tensor, float]],
    n_channels: int,
) -> List[float]:
    """Per-channel weights for a multi-channel similarity metric, scaled to sum to 1.

    None gives every channel the same weight, i.e. the plain channel average.
    Because the weights are normalised, the data term keeps the same scale as a
    single-channel registration, so regularisation weights tuned there carry over.
    A channel with weight 0 is skipped.
    """
    if channel_weights is None:
        return [1.0 / n_channels] * n_channels
    if isinstance(channel_weights, torch.Tensor):
        w = [float(v) for v in channel_weights.reshape(-1).tolist()]
    elif isinstance(channel_weights, (int, float)):
        w = [float(channel_weights)]
    else:
        w = [float(v) for v in channel_weights]
    if len(w) != n_channels:
        raise ValueError(
            f"channel_weights has {len(w)} entries but the images have "
            f"{n_channels} channel(s)")
    if any((not math.isfinite(v)) or v < 0.0 for v in w):
        raise ValueError(f"channel_weights must be finite and >= 0, got {w}")
    total = sum(w)
    if total <= 0.0:
        raise ValueError("channel_weights must not all be zero")
    return [v / total for v in w]
