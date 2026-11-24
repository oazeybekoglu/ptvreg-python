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

## Multi-channel images

Images may have several channels, shape `(1, C, D, H, W)` (or `(1, C, H, W)`
in 2-D), for example several MR sequences or stain channels. Both images
need the same channels in the same order, and one displacement field is
estimated for all of them. With `metric="lcc"` or `metric="mi"` the
similarity is computed per channel pair (channel c of the moving image
against channel c of the fixed image) and combined as a weighted sum:

```python
opts["channel_weights"] = [1.0, 0.5, 0.5]   # one weight per channel
```

The weights are normalized to sum to 1, so the data term keeps the scale of
a single-channel registration and regularization weights tuned there still
apply. Without `channel_weights` all channels count equally, and a weight of
0 leaves that channel out. For MI every channel is binned over the same
`mi_intensity_range` (default `(0, 1)`), so scale each channel to that range.

## Example

`examples/register_pair.py` registers two NIfTI volumes end to end and
writes the voxel-unit displacement field:

```
python examples/register_pair.py fixed.nii.gz moving.nii.gz disp.nii.gz
```

## Changes in 0.3.0

- Multi-channel MI (one joint histogram per channel pair; before, all
  channels were pooled into one histogram, which failed with the default
  border mask) and per-channel weights for LCC and MI (`channel_weights`).
- LCC: the Gaussian window is floored at 0.5 voxel (`lcc_sigma_min_pix`,
  0 disables). With the window given in mm it shrank below one voxel at the
  coarsest pyramid levels, where the metric became noise.
- MI: the joint histogram is accumulated in float64 (`mi_sum_dtype`) and MI
  follows `loss_compute_dtype`. Before, MI ran in the image dtype (float32 by
  default), which biased its value and gradient on large images.
- LCC: the images are cast to `loss_compute_dtype` (float64 by default)
  before the local moments are formed, and the smoothing kernel and voxel
  sum are in float64. Before, products and kernel stayed in the image dtype
  (float32 by default), which cost up to ~1e-4 relative accuracy in the
  loss (variance cancellation in flat regions).
- Symmetric registration (`ic_weight > 0`): `jac_reg_dense` was ignored and
  is now applied to both fields. Objective normalization, L-BFGS tolerances,
  `loss_compute_dtype` and the float64 warp grid now match the one-sided
  solver, so symmetric results differ from 0.2.0.

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
