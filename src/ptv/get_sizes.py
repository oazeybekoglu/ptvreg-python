from typing import Tuple


def get_sizes_from_volsz(sz: Tuple[int, ...]) -> Tuple[Tuple[int, ...], int, int, int]:
    n = len(sz)

    if n == 2:

        volsz = sz
        Nch = 1
        Nd = 2
        Nimgs = 1
    elif n == 3:

        volsz = sz
        Nch = 1
        Nd = 3 if sz[2] > 1 else 2
        Nimgs = 1
    elif n == 4:

        volsz = sz[:3]
        Nd = 2 if sz[2] == 1 else 3
        Nch = sz[3]
        Nimgs = 1
    elif n == 5:

        volsz = sz[:3]
        Nd = 2 if sz[2] == 1 else 3
        Nch = sz[3]
        Nimgs = sz[4]
    else:
        raise ValueError(f"Unexpected size with {n} dimensions")

    return volsz, Nd, Nch, Nimgs
