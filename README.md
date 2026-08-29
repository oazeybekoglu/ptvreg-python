# ptvreg-python

PyTorch implementation of pTVreg: parametric deformable image registration
with isotropic total-variation regularization, solved coarse-to-fine with
ADMM (or L-BFGS). Extended from the original MATLAB
[pTVreg](https://github.com/visva89/pTVreg) by Vishnevskiy et al.

## Installation

```
git clone https://github.com/oazeybekoglu/ptvreg-python
cd ptvreg-python
pip install -e .
```

## Quick start

Images are tensors of shape `(1, 1, D, H, W)`, intensity-normalized to a
comparable range (e.g. clipped HU scaled to `[0, 1]`). The moving image is
registered to the fixed image; the returned displacement is in millimetres
on the fixed grid.

## Example

`examples/register_pair.py` registers two NIfTI volumes end to end and
writes the voxel-unit displacement field:

```
python examples/register_pair.py fixed.nii.gz moving.nii.gz disp.nii.gz
```

## References

If you use this code, please cite:

```
@article{zeybekoglu2026,
  author  = {Zeybekoglu, Onur Ali and Tilly, David and Goksel, Orcun},
  title   = {An Accessible Solution for Deformable Image Registration
             Compared with Learning-Based Approaches},
  journal = {arXiv preprint arXiv:2608.02248},
  year    = {2026},
  url     = {https://arxiv.org/abs/2608.02248}
}
```

## License

Apache 2.0, see `LICENSE`. Adapted from
[pTVreg](https://github.com/visva89/pTVreg); see `NOTICE`.
