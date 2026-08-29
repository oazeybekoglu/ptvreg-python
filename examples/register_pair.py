import sys

import nibabel as nib
import numpy as np
import torch

from ptv import register_pair_min


def load(path, device, dtype):
    nii = nib.load(path)
    img = np.clip(nii.get_fdata(), -1024.0, 0.0)
    img = (img + 1024.0) / 1024.0
    t = torch.as_tensor(img, device=device, dtype=dtype)[None, None]
    return t, nii


def main():
    if len(sys.argv) != 4:
        print("usage: register_pair.py <fixed> <moving> <out_disp>")
        sys.exit(2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    fix_t, fix_nii = load(sys.argv[1], device, dtype)
    mov_t, _ = load(sys.argv[2], device, dtype)
    spacing = tuple(float(z) for z in fix_nii.header.get_zooms()[:3])

    opts = {
        "pix_resolution": spacing,
        "grid_spacing": [8, 8, 8],
        "cp_refinements": 1,
        "k_down": 0.7,
        "metric": "lcc",
        "metric_param": (2.1, 2.1, 2.1),
        "scale_metric_param": True,
        "isoTV": 0.11,
        "csqrt": 5e-3,
        "border_mask": 5,
        "opt_method": "admm",
        "jac_reg": 0.0,
        "jac_reg_dense": 2.0,
        "device": device,
        "verbose": True,
    }

    _voldef, disp_mm, _knots, _infos = register_pair_min(mov_t, fix_t, opts)

    disp = disp_mm[0].detach().cpu().numpy().astype(np.float32)
    for ax in range(3):
        disp[ax] /= spacing[ax]
    nib.save(nib.Nifti1Image(disp, fix_nii.affine), sys.argv[3])
    print(f"wrote {sys.argv[3]}  max|disp| = {np.abs(disp).max():.2f} voxels")


if __name__ == "__main__":
    main()
