import torch
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .get_grid_size import get_grid_size
from .disp_from_knots import disp_from_knots, init_knots
from .params_phys_to_pix import params_phys_to_pix
from .warp_image import warp_image
from .metrics_ssd import ssd_loss, sad_loss
from .metrics_lcc import lcc_loss
from .regularizers_tv import isotropic_tv_regularization
from .regularizers_jac import jac_reg_regularization, jac_reg_regularization_dense
from .upsample_knots import upsample_knots
from .create_pyramid import compute_nlvl
from .inverse_consistency import inverse_consistency_loss


try:
    from .metrics_mi import mi_loss
except ImportError:
    mi_loss = None


@dataclass
class LevelInfo:
    data_loss: float = 0.0
    reg_loss: float = 0.0
    total_loss: float = 0.0
    n_iters: int = 0


    data_loss_norm: float = 0.0
    reg_loss_norm: float = 0.0
    total_loss_norm: float = 0.0
    oscale: float = 1.0
    admm_history: dict = None


    data_loss_inv: float = 0.0
    ic_loss: float = 0.0
    reg_loss_inv: float = 0.0


    knots_fm: object = None
    knots_mf: object = None
    grid_spacing: tuple = None
    image_size: tuple = None


def _emit_level_log(opts, level_num, total_levels, info, image_size, grid_spacing):
    log_fn = opts.get("log_fn")
    if log_fn is None:
        return
    try:
        log_fn({
            "level": int(level_num),
            "total_levels": int(total_levels),
            "data_loss": float(info.data_loss),
            "reg_loss": float(info.reg_loss),
            "total_loss": float(info.total_loss),
            "n_iters": int(info.n_iters),
            "grid_spacing": tuple(grid_spacing) if grid_spacing else None,
            "image_size": tuple(image_size) if image_size else None,
            "admm_history": info.admm_history,
        })
    except Exception as e:
        if opts.get("verbose"):
            print(f"  [log_fn] warning: logging failed: {e}")


def _resolve_loss_dtype(val):
    if val is None or isinstance(val, torch.dtype):
        return val
    return {"float64": torch.float64, "float32": torch.float32,
            "fp64": torch.float64, "fp32": torch.float32,
            "double": torch.float64, "single": torch.float32}.get(str(val).lower())


def _loss_settings(opts, model_dtype):
    compute_dtype = _resolve_loss_dtype(opts.get("loss_compute_dtype", torch.float64))
    normalize = bool(opts.get("normalize_objective", True))


    tol_grad = float(opts.get("lbfgs_tol_grad", 1e-8))
    tol_change = float(opts.get("lbfgs_tol_change", 1e-9))
    return compute_dtype, normalize, tol_grad, tol_change


def _obj_scale(normalize, image_size, pix_res):
    if not normalize:
        return 1.0
    n_vox = 1
    for s in image_size:
        n_vox *= int(s)
    pv = 1.0
    for p in pix_res:
        pv *= float(p)
    denom = n_vox * pv
    return 1.0 / denom if denom > 0 else 1.0


LCC_SIGMA_MIN_PIX_DEFAULT = 0.5


def _lcc_sigma_pix(opts, pix_res_cur, nd):
    """LCC window sigma in voxels of the current pyramid level.

    With scale_metric_param=True (default) metric_param is a physical sigma
    in mm, so the window measured in voxels shrinks at coarse levels. Below
    ~0.5 voxel nearly all window weight sits on the centre voxel, the local
    variance vanishes and the metric becomes epsilon noise at those levels.
    lcc_sigma_min_pix (default 0.5) floors the sigma in voxels; it only
    changes the coarsest levels. Pass 0 or None to disable.
    """
    mp = opts.get("metric_param", (7.0,) * nd)
    if opts.get("scale_metric_param", True):
        sigma_pix = tuple(mp[d] / pix_res_cur[d] for d in range(nd))
    else:
        pr0 = opts.get("pix_resolution", (1.0,) * nd)
        sigma_pix = tuple(mp[d] / pr0[d] for d in range(nd))
    smin = opts.get("lcc_sigma_min_pix", LCC_SIGMA_MIN_PIX_DEFAULT)
    if smin:
        sigma_pix = tuple(max(sp, float(smin)) for sp in sigma_pix)
    return sigma_pix


def _dense_jac_penalty(d, opts):
    """jac_reg_dense * dense Jacobian folding penalty of one field (0.0 if off)."""
    w = float(opts.get("jac_reg_dense", 0.0))
    if w <= 0.0:
        return 0.0
    return w * jac_reg_regularization_dense(
        d,
        lower_bound=float(opts.get("jac_dense_lb", 0.5)),
        upper_bound=float(opts.get("jac_dense_ub", 4.0)))


def _build_data_loss(warped, fixed, opts, pix_res_cur, mask, compute_dtype=None):
    metric = opts["metric"]
    nd = warped.dim() - 2

    if metric == "ssd":
        return ssd_loss(warped, fixed, pix_res_cur, mask)

    elif metric == "sad":
        return sad_loss(warped, fixed, pix_res_cur, mask)

    elif metric == "lcc":
        sigma_pix = _lcc_sigma_pix(opts, pix_res_cur, nd)

        eps = opts.get("deps", 1e-4)
        use_abs = opts.get("loc_cc_abs", False)
        return lcc_loss(warped, fixed, pix_res_cur, sigma_pix,
                        mask=mask, eps=eps, use_abs=use_abs,
                        compute_dtype=compute_dtype,
                        return_dtype=_resolve_loss_dtype(
                            opts.get("lcc_return_dtype", None)),
                        channel_weights=opts.get("channel_weights", None))

    elif metric == "mi":


        mp = opts.get("metric_param", (32,))
        if hasattr(mp, "__len__"):
            n_bins = int(mp[0])
        else:
            n_bins = int(mp)
        if mi_loss is None:
            raise RuntimeError("metric='mi' requested but metrics_mi.py is "
                               "not available in this build (LCC-only bundle).")
        intensity_range = opts.get("mi_intensity_range", (0.0, 1.0))
        eps = opts.get("mi_eps", 1e-10)
        min_vpb = opts.get("mi_min_voxels_per_bin", 50.0)
        mi_verbose = bool(opts.get("verbose", False))
        return mi_loss(warped, fixed, pix_res_cur,
                       n_bins=n_bins, intensity_range=intensity_range,
                       mask=mask, eps=eps,
                       min_voxels_per_bin=min_vpb, verbose=mi_verbose,
                       compute_dtype=compute_dtype,
                       sum_dtype=_resolve_loss_dtype(
                           opts.get("mi_sum_dtype", torch.float64)),
                       return_dtype=_resolve_loss_dtype(
                           opts.get("mi_return_dtype",
                                    opts.get("lcc_return_dtype", None))),
                       channel_weights=opts.get("channel_weights", None))

    else:
        raise NotImplementedError(f"Unsupported metric='{metric}'")


def _build_border_mask(spatial_size, border, device, dtype):
    nd = len(spatial_size)
    if border <= 0:
        return None
    mask = torch.ones((1, 1, *spatial_size), device=device, dtype=dtype)
    for d in range(nd):
        slices_lo = [slice(None)] * (2 + nd)
        slices_hi = [slice(None)] * (2 + nd)
        slices_lo[2 + d] = slice(0, border)
        slices_hi[2 + d] = slice(spatial_size[d] - border, None)
        mask[tuple(slices_lo)] = 0
        mask[tuple(slices_hi)] = 0
    return mask


def _gaussian_blur(image, sigma, nd):
    if sigma < 0.1:
        return image
    ksz = int(2 * math.ceil(2 * sigma) + 1)
    ksz = max(ksz, 5)
    half = ksz // 2

    x = torch.arange(-half, half + 1, device=image.device, dtype=image.dtype)
    g1d = torch.exp(-0.5 * (x / sigma) ** 2)
    g1d = g1d / g1d.sum()

    C = image.shape[1]
    if nd == 2:
        kernel = (g1d.unsqueeze(1) * g1d.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
        padded = F.pad(image, [half] * 4, mode='reflect')
        return F.conv2d(padded, kernel.expand(C, -1, -1, -1), groups=C)
    else:
        kernel = (g1d.reshape(-1, 1, 1) * g1d.reshape(1, -1, 1)
                  * g1d.reshape(1, 1, -1)).unsqueeze(0).unsqueeze(0)
        padded = F.pad(image, [half] * 6, mode='reflect')
        return F.conv3d(padded, kernel.expand(C, -1, -1, -1, -1), groups=C)


def _my_gradient_diff(x: torch.Tensor, h: float, dim: int) -> torch.Tensor:
    n = x.shape[dim]
    if n <= 1:
        return torch.zeros_like(x)

    fwd = (x.narrow(dim, 1, n - 1) - x.narrow(dim, 0, n - 1)) / h
    last = fwd.narrow(dim, n - 2, 1)
    return torch.cat([fwd, last], dim=dim)


def _compute_D_knots(
    knots: torch.Tensor,
    spacing: Tuple[float, ...],
    nd: int,
    differentiable: bool = False,
) -> torch.Tensor:
    from .regularizers_tv import _my_gradient

    grad_fn = _my_gradient_diff if differentiable else _my_gradient
    Nd = knots.shape[1]
    components = []
    for comp in range(Nd):
        for d in range(nd):
            g = grad_fn(knots[:, comp : comp + 1], spacing[d], dim=d + 2)
            components.append(g)
    return torch.cat(components, dim=1)


def _my_gradient_adjoint(v: torch.Tensor, h: float, dim: int) -> torch.Tensor:
    n = v.shape[dim]
    ndim = v.dim()

    if n <= 1:
        return torch.zeros_like(v)

    def _sl(d, s):
        slices = [slice(None)] * ndim
        slices[d] = s
        return tuple(slices)

    result = torch.zeros_like(v)

    if n == 2:
        result[_sl(dim, 0)] = (-v[_sl(dim, 0)] - v[_sl(dim, 1)]) / h
        result[_sl(dim, 1)] = (v[_sl(dim, 0)] + v[_sl(dim, 1)]) / h
        return result


    result[_sl(dim, 0)] = -v[_sl(dim, 0)] / h


    if n > 3:
        result[_sl(dim, slice(1, n - 2))] = (
            v[_sl(dim, slice(0, n - 3))] - v[_sl(dim, slice(1, n - 2))]
        ) / h


    result[_sl(dim, n - 2)] = (
        v[_sl(dim, n - 3)] - v[_sl(dim, n - 2)] - v[_sl(dim, n - 1)]
    ) / h


    result[_sl(dim, n - 1)] = (
        v[_sl(dim, n - 2)] + v[_sl(dim, n - 1)]
    ) / h

    return result


def _D_adjoint(
    V: torch.Tensor,
    nd: int,
    spacing: Tuple[float, ...],
) -> torch.Tensor:
    Nd = nd
    batch = V.shape[0]
    ksz = V.shape[2:]
    result = torch.zeros(batch, Nd, *ksz, device=V.device, dtype=V.dtype)

    for comp in range(Nd):
        for d in range(nd):
            ch = comp * nd + d
            adj = _my_gradient_adjoint(
                V[:, ch : ch + 1], spacing[d], dim=d + 2
            )
            result[:, comp : comp + 1] += adj

    return result


def _block_soft_threshold(
    A: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    norm = torch.sqrt((A ** 2).sum(dim=1, keepdim=True))
    scale = (1.0 - threshold / (norm + 1e-30)).clamp(min=0.0)
    return scale * A


def _optimize_level(
    knots, vol_mov, vol_fix, opts,
    cur_imsz, cur_pr, cur_gs, nd,
    isoTV, csqrt, border_mask, interp_type, spline_order,
    max_iters, opt_method, verbose, device, dtype,
    level_label="",
):

    mask = _build_border_mask(cur_imsz, border_mask, device, dtype)
    tv_spacing = tuple(cur_pr[d] * cur_gs[d] for d in range(nd))


    _cdt, _normalize, _tol_grad, _tol_change = _loss_settings(opts, dtype)
    _oscale = _obj_scale(_normalize, cur_imsz, cur_pr)


    _wgdt = _resolve_loss_dtype(opts.get("warp_compute_dtype", torch.float64))

    best_knots = knots.detach().clone()


    if opt_method == "admm":
        return _optimize_level_admm(
            knots, vol_mov, vol_fix, opts,
            cur_imsz, cur_pr, cur_gs, nd,
            isoTV, csqrt, border_mask, interp_type, spline_order,
            max_iters, verbose, device, dtype, mask, tv_spacing,
        )


    if opt_method == "lbfgs":


        n_knot_elements = knots.numel()
        if n_knot_elements > 100**3:
            hist_size = 20
        elif n_knot_elements > 50**3:
            hist_size = 30
        else:
            hist_size = 85

        if verbose:
            print(f"  L-BFGS: {n_knot_elements} knot elements, history_size={hist_size}")

        optimizer = torch.optim.LBFGS(
            [knots],
            max_iter=max_iters,
            history_size=hist_size,
            line_search_fn="strong_wolfe",
            lr=1.0,


            tolerance_grad=_tol_grad * _oscale,
            tolerance_change=_tol_change * _oscale,
        )

        iter_count = [0]
        last_loss = [float("inf")]

        def closure():
            optimizer.zero_grad()
            kp = params_phys_to_pix(knots, cur_pr, spline_order)
            d = disp_from_knots(kp, cur_imsz, cur_gs, spline_order)
            w = warp_image(vol_mov, d, interp_type, grid_dtype=_wgdt)
            data_loss = _build_data_loss(w, vol_fix, opts, cur_pr, mask,
                                         compute_dtype=_cdt)


            reg_loss = torch.tensor(0.0, device=device, dtype=dtype)
            if isoTV > 0:
                reg_loss = reg_loss + isoTV * isotropic_tv_regularization(
                    knots, tv_spacing, csqrt, compute_dtype=_cdt)
            _jac_reg = float(opts.get("jac_reg", 0.0))
            if _jac_reg > 0:
                reg_loss = reg_loss + _jac_reg * jac_reg_regularization(knots, tv_spacing)


            _jac_dense = float(opts.get("jac_reg_dense", 0.0))
            if _jac_dense > 0:
                reg_loss = reg_loss + _jac_dense * jac_reg_regularization_dense(
                    d,
                    lower_bound=float(opts.get("jac_dense_lb", 0.5)),
                    upper_bound=float(opts.get("jac_dense_ub", 4.0)))
            total = (data_loss + reg_loss) * _oscale
            total.backward()

            last_loss[0] = total.item()
            iter_count[0] += 1
            if verbose and (iter_count[0] % 20 == 0):
                print(f"  iter {iter_count[0]:4d}: loss = {total.item():.6f}")

            _ilog = opts.get("iter_log_fn")
            if _ilog is not None:
                try:
                    _ilog({
                        "phase": "lbfgs",
                        "level": opts.get("_cur_level"),
                        "iter": iter_count[0],
                        "data_loss": float(data_loss.item()),
                        "reg_loss": float(reg_loss.item())
                            if torch.is_tensor(reg_loss) else float(reg_loss),
                        "total_loss": float(total.item()),
                    })
                except Exception:
                    pass

            return total

        optimizer.step(closure)
        best_knots = knots.detach().clone()

        if verbose:
            print(f"  final: loss = {last_loss[0]:.6f} ({iter_count[0]} func evals)")

        n_iters = iter_count[0]


    elif opt_method == "adam":
        best_loss = float("inf")
        optimizer = torch.optim.Adam([knots], lr=0.05)

        for it in range(max_iters):
            optimizer.zero_grad()
            kp = params_phys_to_pix(knots, cur_pr, spline_order)
            d = disp_from_knots(kp, cur_imsz, cur_gs, spline_order)
            w = warp_image(vol_mov, d, interp_type, grid_dtype=_wgdt)
            data_loss = _build_data_loss(w, vol_fix, opts, cur_pr, mask,
                                         compute_dtype=_cdt)


            reg_loss = torch.tensor(0.0, device=device, dtype=dtype)
            if isoTV > 0:
                reg_loss = reg_loss + isoTV * isotropic_tv_regularization(
                    knots, tv_spacing, csqrt, compute_dtype=_cdt)
            _jac_reg = float(opts.get("jac_reg", 0.0))
            if _jac_reg > 0:
                reg_loss = reg_loss + _jac_reg * jac_reg_regularization(knots, tv_spacing)

            _jac_dense = float(opts.get("jac_reg_dense", 0.0))
            if _jac_dense > 0:
                reg_loss = reg_loss + _jac_dense * jac_reg_regularization_dense(
                    d,
                    lower_bound=float(opts.get("jac_dense_lb", 0.5)),
                    upper_bound=float(opts.get("jac_dense_ub", 4.0)))
            total = (data_loss + reg_loss) * _oscale
            total.backward()
            optimizer.step()

            lv = total.item()
            if lv < best_loss:
                best_loss = lv
                best_knots = knots.detach().clone()
            if verbose and (it % 20 == 0 or it == max_iters - 1):
                print(f"  iter {it:4d}: loss = {lv:.6f}")

        n_iters = max_iters

    else:
        raise ValueError(f"Unknown opt_method='{opt_method}'")


    knots = best_knots.clone().requires_grad_(True)


    with torch.no_grad():
        kp = params_phys_to_pix(knots, cur_pr, spline_order)
        d = disp_from_knots(kp, cur_imsz, cur_gs, spline_order)
        w = warp_image(vol_mov, d, interp_type, grid_dtype=_wgdt)
        final_data = _build_data_loss(w, vol_fix, opts, cur_pr, mask).item()
        final_reg = 0.0
        if isoTV > 0:
            final_reg += (isoTV * isotropic_tv_regularization(
                knots, tv_spacing, csqrt)).item()
        if float(opts.get("jac_reg", 0)) > 0:
            final_reg += (float(opts.get("jac_reg", 0)) * jac_reg_regularization(knots, tv_spacing)).item()
        if float(opts.get("jac_reg_dense", 0)) > 0:
            final_reg += (float(opts.get("jac_reg_dense", 0)) * jac_reg_regularization_dense(
                d,
                lower_bound=float(opts.get("jac_dense_lb", 0.5)),
                upper_bound=float(opts.get("jac_dense_ub", 4.0)))).item()

    info = LevelInfo(
        data_loss=final_data,
        reg_loss=final_reg,
        total_loss=final_data + final_reg,
        n_iters=n_iters,
        data_loss_norm=final_data * _oscale,
        reg_loss_norm=final_reg * _oscale,
        total_loss_norm=(final_data + final_reg) * _oscale,
        oscale=_oscale,
    )

    return knots, info


def _optimize_level_admm(
    knots, vol_mov, vol_fix, opts,
    cur_imsz, cur_pr, cur_gs, nd,
    isoTV, csqrt, border_mask_px, interp_type, spline_order,
    max_iters, verbose, device, dtype,
    mask, tv_spacing,
):
    lam = isoTV
    eta = 1.0
    for s in tv_spacing[:nd]:
        eta *= s


    _cdt, _normalize, _tol_grad, _tol_change = _loss_settings(opts, dtype)
    _oscale = _obj_scale(_normalize, cur_imsz, cur_pr)


    _wgdt = _resolve_loss_dtype(opts.get("warp_compute_dtype", torch.float64))
    admm_max_outer = int(opts.get("admm_max_outer", 10))
    lbfgs_iters = int(opts.get("admm_lbfgs_iters",30))
    mu = float(opts.get("admm_mu", 10.0))
    tau = float(opts.get("admm_tau", 1.3))
    tol = float(opts.get("admm_tol", 1e-2))


    rho_user = opts.get("admm_rho", None)
    if rho_user is not None and float(rho_user) > 0:
        rho = float(rho_user)
    else:
        rho = max(lam * eta, 1.0)

    if verbose:
        threshold_init = lam * eta / rho
        print(f"    ADMM: rho0={rho:.2f} (lam*eta={lam*eta:.2f}), "
              f"threshold0={threshold_init:.4f}, mu={mu}, tau={tau}, "
              f"outer={admm_max_outer}, inner_lbfgs={lbfgs_iters}")


    with torch.no_grad():
        Dk = _compute_D_knots(knots, tv_spacing, nd, differentiable=False)
    Z = Dk.clone()
    U = torch.zeros_like(Dk)


    knots_opt = knots.detach().clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [knots_opt],
        max_iter=lbfgs_iters,
        history_size=5,
        line_search_fn="strong_wolfe",
        lr=1.0,


        tolerance_grad=_tol_grad * _oscale,
        tolerance_change=_tol_change * _oscale,
    )

    total_func_evals = 0
    knots_prev = knots_opt.detach().clone()


    admm_hist = {
        'data': [], 'penalty': [], 'tv': [], 'total': [],
        'rho': [], 'threshold': [],
        'r_norm': [], 's_norm': [],
        'dk': [], 'z_sparse_frac': [],
    }

    for admm_it in range(admm_max_outer):


        target = (Z - U).detach()
        cur_rho = rho

        func_evals = [0]

        def closure():
            optimizer.zero_grad()


            kp = params_phys_to_pix(knots_opt, cur_pr, spline_order)
            d = disp_from_knots(kp, cur_imsz, cur_gs, spline_order)
            w = warp_image(vol_mov, d, interp_type, grid_dtype=_wgdt)
            data_loss = _build_data_loss(w, vol_fix, opts, cur_pr, mask,
                                         compute_dtype=_cdt)


            Dk_cur = _compute_D_knots(
                knots_opt, tv_spacing, nd, differentiable=True
            )
            penalty = (cur_rho / 2.0) * ((Dk_cur - target) ** 2).sum()


            _jac_reg = float(opts.get("jac_reg", 0.0))
            if _jac_reg > 0:
                penalty = penalty + _jac_reg * jac_reg_regularization(
                    knots_opt, tv_spacing)


            _jac_dense = float(opts.get("jac_reg_dense", 0.0))
            if _jac_dense > 0:
                penalty = penalty + _jac_dense * jac_reg_regularization_dense(
                    d,
                    lower_bound=float(opts.get("jac_dense_lb", 0.5)),
                    upper_bound=float(opts.get("jac_dense_ub", 4.0)))

            total = data_loss + penalty
            total = total * _oscale
            total.backward()

            func_evals[0] += 1
            return total

        optimizer.step(closure)
        total_func_evals += func_evals[0]


        if verbose:
            with torch.no_grad():
                kp_k = params_phys_to_pix(knots_opt, cur_pr, spline_order)
                d_k = disp_from_knots(kp_k, cur_imsz, cur_gs, spline_order)
                w_k = warp_image(vol_mov, d_k, interp_type, grid_dtype=_wgdt)
                kdata = _build_data_loss(w_k, vol_fix, opts, cur_pr, mask).item()
                Dk_tmp = _compute_D_knots(knots_opt, tv_spacing, nd, differentiable=False)
                kpenalty = (cur_rho / 2.0) * ((Dk_tmp - target) ** 2).sum().item()
            print(f"      k-update: data={kdata:.2f}  penalty={kpenalty:.2f}  "
                  f"total={kdata+kpenalty:.2f}  ({func_evals[0]} fevals)"
                  f"  | norm: data={kdata*_oscale:.6f} "
                  f"penalty={kpenalty*_oscale:.6f} "
                  f"total={(kdata+kpenalty)*_oscale:.6f}")


        with torch.no_grad():
            Dk = _compute_D_knots(
                knots_opt, tv_spacing, nd, differentiable=False
            )


        Z_old = Z.clone()
        A = Dk + U
        threshold = lam * eta / rho
        Z = _block_soft_threshold(A, threshold)

        if verbose:
            with torch.no_grad():
                a_norms = torch.sqrt((A ** 2).sum(dim=1))
                z_norms_upd = torch.sqrt((Z ** 2).sum(dim=1))
                n_zeroed = (z_norms_upd < 1e-10).sum().item()
                n_total = z_norms_upd.numel()
            print(f"      Z-update: threshold={threshold:.4f}  "
                  f"zeroed={n_zeroed}/{n_total} ({100*n_zeroed/max(1,n_total):.1f}%)")


        U = U + Dk - Z


        r = Dk - Z
        with torch.no_grad():
            s = rho * _D_adjoint(Z - Z_old, nd, tv_spacing)

        r_norm = r.double().norm().item()
        s_norm = s.double().norm().item()

        rho_old = rho
        if r_norm >= mu * s_norm:
            rho = tau * rho
            U = U / tau
        elif s_norm >= mu * r_norm:
            rho = rho / tau
            U = U * tau

        if verbose and rho != rho_old:
            direction = "UP" if rho > rho_old else "DOWN"
            print(f"      rho: {rho_old:.2f} -> {rho:.2f} ({direction})")


        with torch.no_grad():
            k_change = (knots_opt - knots_prev).abs().max().item()
            knots_prev = knots_opt.detach().clone()


        cur_threshold = lam * eta / rho
        with torch.no_grad():
            kp_mon = params_phys_to_pix(knots_opt, cur_pr, spline_order)
            d_mon = disp_from_knots(kp_mon, cur_imsz, cur_gs, spline_order)
            w_mon = warp_image(vol_mov, d_mon, interp_type, grid_dtype=_wgdt)
            data_loss_mon = _build_data_loss(w_mon, vol_fix, opts, cur_pr, mask).item()

            Dk_hist = _compute_D_knots(knots_opt, tv_spacing, nd, differentiable=False)
            penalty_mon = (rho / 2.0) * ((Dk_hist - (Z - U + (Dk - Z))) ** 2).sum().item()

            penalty_mon = (rho / 2.0) * (r ** 2).sum().item()

            z_norms = torch.sqrt((Z ** 2).sum(dim=1))
            tv_val = (lam * eta * z_norms.sum() / knots_opt.shape[0]).item()
            z_zero_frac = (z_norms < 1e-10).float().mean().item()
            total_mon = data_loss_mon + tv_val


        admm_hist['data'].append(data_loss_mon)
        admm_hist['penalty'].append(penalty_mon)
        admm_hist['tv'].append(tv_val)
        admm_hist['total'].append(total_mon)
        admm_hist['rho'].append(rho)
        admm_hist['threshold'].append(cur_threshold)
        admm_hist['r_norm'].append(r_norm)
        admm_hist['s_norm'].append(s_norm)
        admm_hist['dk'].append(k_change)
        admm_hist['z_sparse_frac'].append(z_zero_frac)

        _ilog = opts.get("iter_log_fn")
        if _ilog is not None:
            try:
                _ilog({
                    "phase": "admm",
                    "level": opts.get("_cur_level"),
                    "iter": admm_it + 1,
                    "data_loss": float(data_loss_mon),
                    "reg_loss": float(tv_val),
                    "total_loss": float(total_mon),
                    "rho": float(rho),
                    "threshold": float(cur_threshold),
                    "r_norm": float(r_norm),
                    "s_norm": float(s_norm),
                })
            except Exception:
                pass

        if verbose:
            print(f"    ADMM iter {admm_it+1:3d}: "
                  f"data={data_loss_mon:.2f}  tv={tv_val:.2f}  total={total_mon:.2f}  "
                  f"r={r_norm:.4e}  s={s_norm:.4e}  "
                  f"rho={rho:.2f}  thr={cur_threshold:.4f}  "
                  f"dk={k_change:.4e}  Z_sparse={z_zero_frac:.1%}  "
                  f"fevals={func_evals[0]}")
            print(f"              norm: data={data_loss_mon*_oscale:.6f}  "
                  f"tv={tv_val*_oscale:.6f}  "
                  f"total={total_mon*_oscale:.6f}")

        if k_change < tol and admm_it > 0:
            if verbose:
                print(f"    ADMM converged at iter {admm_it+1} "
                      f"(dk={k_change:.2e} < tol={tol:.2e})")
            break


    with torch.no_grad():
        kp = params_phys_to_pix(knots_opt, cur_pr, spline_order)
        d = disp_from_knots(kp, cur_imsz, cur_gs, spline_order)
        w = warp_image(vol_mov, d, interp_type, grid_dtype=_wgdt)
        final_data = _build_data_loss(w, vol_fix, opts, cur_pr, mask).item()


        z_norms = torch.sqrt((Z ** 2).sum(dim=1))
        final_reg = (lam * eta * z_norms.sum() / knots_opt.shape[0]).item()
        _jac_reg = float(opts.get("jac_reg", 0.0))
        if _jac_reg > 0:
            final_reg += (_jac_reg * jac_reg_regularization(
                knots_opt, tv_spacing)).item()
        if float(opts.get("jac_reg_dense", 0)) > 0:
            final_reg += (float(opts.get("jac_reg_dense", 0)) * jac_reg_regularization_dense(
                d,
                lower_bound=float(opts.get("jac_dense_lb", 0.5)),
                upper_bound=float(opts.get("jac_dense_ub", 4.0)))).item()

    knots_out = knots_opt.detach().clone().requires_grad_(True)

    info = LevelInfo(
        data_loss=final_data,
        reg_loss=final_reg,
        total_loss=final_data + final_reg,
        n_iters=total_func_evals,
        admm_history=admm_hist,
        data_loss_norm=final_data * _oscale,
        reg_loss_norm=final_reg * _oscale,
        total_loss_norm=(final_data + final_reg) * _oscale,
        oscale=_oscale,
    )

    if verbose:
        print(f"  ADMM done: data={final_data:.4f} reg={final_reg:.4f} "
              f"total={final_data+final_reg:.4f} ({total_func_evals} fevals)")
        print(f"    normalized: data={final_data*_oscale:.6f} "
              f"reg={final_reg*_oscale:.6f} "
              f"total={(final_data+final_reg)*_oscale:.6f} (oscale={_oscale:.3e})")

    return knots_out, info


def render_level_field_full_res(
    knots_phys,
    level_grid_spacing,
    level_image_size,
    full_image_size,
    pix_resolution,
    spline_order=1,
    device=None,
    dtype=None,
):
    if device is None:
        device = knots_phys.device
    if dtype is None:
        dtype = knots_phys.dtype
    nd = knots_phys.dim() - 2
    k = knots_phys.to(device=device, dtype=dtype)


    if tuple(level_image_size) != tuple(full_image_size):
        k = upsample_knots(
            k, level_image_size, level_grid_spacing,
            full_image_size, level_grid_spacing,
            spline_order=spline_order, method="direct",
        )

    with torch.no_grad():
        kp = params_phys_to_pix(k, pix_resolution, spline_order)
        disp_pix = disp_from_knots(kp, full_image_size, level_grid_spacing,
                                   spline_order)
        pr_t = torch.tensor(pix_resolution[:nd], device=device, dtype=dtype)
        pr_t = pr_t.view(1, nd, *([1] * nd))
        Tmm = disp_pix * pr_t
    return Tmm


def register_pair_min(
    mov: torch.Tensor,
    fix: torch.Tensor,
    opts: Dict,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[LevelInfo]]:


    if float(opts.get("ic_weight", 0.0)) > 0:


        voldef_FM, _, Tmm_FM, _, Kmm_FM, _, infos = register_pair_symmetric(
            mov, fix, opts
        )
        return voldef_FM, Tmm_FM, Kmm_FM, infos


    device_name = opts.get("device", "cpu")
    dtype = opts.get("dtype", torch.float32)
    device = torch.device(device_name)

    mov = mov.to(device=device, dtype=dtype)
    fix = fix.to(device=device, dtype=dtype)

    nd = mov.dim() - 2
    spatial_size = tuple(mov.shape[2:])

    grid_spacing = tuple(opts.get("grid_spacing", (4,) * nd))[:nd]
    pix_resolution = tuple(opts.get("pix_resolution", (1.0,) * nd))[:nd]
    spline_order = int(opts.get("spline_order", 1))
    k_down = float(opts.get("k_down", 0.7))
    max_iters = int(opts.get("max_iters", 100))
    isoTV = float(opts.get("isoTV", 0.0))
    csqrt = float(opts.get("csqrt", 5e-3))
    border_mask = int(opts.get("border_mask", 3))
    interp_type = int(opts.get("interp_type", 0))
    verbose = bool(opts.get("verbose", False))
    opt_method = str(opts.get("opt_method", "lbfgs")).lower()
    cp_refinements = int(opts.get("cp_refinements", 0))


    metric = str(opts.get("metric", "ssd"))
    if metric in ("loc_cc_fftn", "loc_cc_fftn_gpu", "loc_cc_fftn_single",
                   "loc_cc_fftn_gpu_single"):
        metric = "lcc"
    opts["metric"] = metric


    nlvl_raw = opts.get("nlvl", None)
    if nlvl_raw is None or (hasattr(nlvl_raw, '__len__') and len(nlvl_raw) == 0):
        nlvl = compute_nlvl(spatial_size, grid_spacing, k_down)
    else:
        nlvl = int(nlvl_raw)
    nlvl = max(1, nlvl)

    if verbose:
        total_levels = nlvl + cp_refinements
        print(f"Registration: {nlvl} pyramid levels + {cp_refinements} "
              f"cp_refinements = {total_levels} total, "
              f"opt_method={opt_method}")


    sigma_blur = 0.4 * (0.5 / k_down)
    mode = "bilinear" if nd == 2 else "trilinear"

    mov_pyr = [mov]
    fix_pyr = [fix]
    pix_res_pyr = [pix_resolution]

    cur_mov = mov
    cur_fix = fix
    cur_pr = pix_resolution

    for lvl in range(1, nlvl):
        cur_mov = _gaussian_blur(cur_mov, sigma_blur, nd)
        cur_fix = _gaussian_blur(cur_fix, sigma_blur, nd)
        new_size = tuple(math.ceil(k_down * s) for s in cur_mov.shape[2:])
        cur_mov = F.interpolate(cur_mov, size=new_size, mode=mode, align_corners=True)
        cur_fix = F.interpolate(cur_fix, size=new_size, mode=mode, align_corners=True)
        cur_pr = tuple(p / k_down for p in cur_pr)
        mov_pyr.append(cur_mov)
        fix_pyr.append(cur_fix)
        pix_res_pyr.append(cur_pr)


    knots = None
    prev_imsz = None
    prev_gs = None
    cur_gs = grid_spacing
    infos = []

    total_levels = nlvl + cp_refinements

    for lvl_idx in range(nlvl - 1, -1, -1):
        vol_mov = mov_pyr[lvl_idx]
        vol_fix = fix_pyr[lvl_idx]
        cur_imsz = tuple(vol_fix.shape[2:])
        cur_pr = pix_res_pyr[lvl_idx]
        cur_gs = grid_spacing


        level_num = nlvl - lvl_idx
        if verbose:
            print(f"Level {level_num}/{total_levels}: "
                  f"size={cur_imsz}, grid={cur_gs}, "
                  f"pix_res={tuple(f'{p:.3f}' for p in cur_pr)}"
                  + (f", lcc_sigma_vox={tuple(f'{v:.2f}' for v in _lcc_sigma_pix(opts, cur_pr, nd))}"
                     if opts.get("metric") == "lcc" else ""))

        if knots is None:
            knots = init_knots(
                cur_imsz, cur_gs, nd, 1,
                spline_order, device=str(device), dtype=dtype,
            )
        else:
            knots = upsample_knots(
                knots, prev_imsz, prev_gs,
                cur_imsz, cur_gs,
                spline_order=spline_order,
                method="direct",
            ).detach().clone().to(device=device, dtype=dtype).requires_grad_(True)

        opts["_cur_level"] = level_num
        opts["_total_levels"] = total_levels
        knots, info = _optimize_level(
            knots, vol_mov, vol_fix, opts,
            cur_imsz, cur_pr, cur_gs, nd,
            isoTV, csqrt, border_mask, interp_type, spline_order,
            max_iters, opt_method, verbose, device, dtype,
        )
        info.knots_fm = knots.detach().cpu().clone()
        info.grid_spacing = tuple(cur_gs)
        info.image_size = tuple(cur_imsz)
        infos.append(info)
        _emit_level_log(opts, level_num, total_levels, info, cur_imsz, cur_gs)

        prev_imsz = cur_imsz
        prev_gs = cur_gs


    for ref_idx in range(cp_refinements):
        vol_mov = mov_pyr[0]
        vol_fix = fix_pyr[0]
        cur_imsz = tuple(vol_fix.shape[2:])
        cur_pr = pix_res_pyr[0]


        old_gs = cur_gs
        cur_gs = tuple(round(g / 2) for g in cur_gs)

        level_num = nlvl + ref_idx + 1
        if verbose:
            print(f"Level {level_num}/{total_levels} (cp_refine {ref_idx+1}): "
                  f"size={cur_imsz}, grid={old_gs}->{cur_gs}, "
                  f"pix_res={tuple(f'{p:.3f}' for p in cur_pr)}")

        knots = upsample_knots(
            knots, prev_imsz, old_gs,
            cur_imsz, cur_gs,
            spline_order=spline_order,
            method="direct",
        ).detach().clone().to(device=device, dtype=dtype).requires_grad_(True)

        if verbose:
            ksz_new = get_grid_size(cur_imsz[:nd], cur_gs[:nd], spline_order)
            n_params = 1
            for k in ksz_new:
                n_params *= k
            n_params *= nd
            print(f"  knots: {ksz_new}, {n_params} parameters")

        opts["_cur_level"] = level_num
        opts["_total_levels"] = total_levels
        knots, info = _optimize_level(
            knots, vol_mov, vol_fix, opts,
            cur_imsz, cur_pr, cur_gs, nd,
            isoTV, csqrt, border_mask, interp_type, spline_order,
            max_iters, opt_method, verbose, device, dtype,
        )
        info.knots_fm = knots.detach().cpu().clone()
        info.grid_spacing = tuple(cur_gs)
        info.image_size = tuple(cur_imsz)
        infos.append(info)
        _emit_level_log(opts, level_num, total_levels, info, cur_imsz, cur_gs)

        prev_imsz = cur_imsz
        prev_gs = cur_gs


    with torch.no_grad():
        kp = params_phys_to_pix(knots, pix_resolution, spline_order)
        disp_pix = disp_from_knots(kp, spatial_size, cur_gs, spline_order)
        voldef = warp_image(mov, disp_pix, interp_type)

        pr_t = torch.tensor(pix_resolution, device=device, dtype=dtype)
        pr_t = pr_t.view(1, nd, *([1] * nd))
        Tmm = disp_pix * pr_t

    return voldef, Tmm, knots.detach(), infos


def _per_direction_reg(knots, opts, tv_spacing, csqrt, isoTV):
    reg = torch.tensor(0.0, device=knots.device, dtype=knots.dtype)
    if isoTV > 0:
        reg = reg + isoTV * isotropic_tv_regularization(knots, tv_spacing, csqrt)
    _jac_reg = float(opts.get("jac_reg", 0.0))
    if _jac_reg > 0:
        reg = reg + _jac_reg * jac_reg_regularization(knots, tv_spacing)
    return reg


def _optimize_level_symmetric(
    knots_fm, knots_mf, vol_mov, vol_fix, opts,
    cur_imsz, cur_pr, cur_gs, nd,
    isoTV, csqrt, border_mask, interp_type, spline_order,
    max_iters, opt_method, ic_weight, verbose, device, dtype,
):
    if opt_method == "admm":
        return _optimize_level_admm_symmetric(
            knots_fm, knots_mf, vol_mov, vol_fix, opts,
            cur_imsz, cur_pr, cur_gs, nd,
            isoTV, csqrt, border_mask, interp_type, spline_order,
            max_iters, ic_weight, verbose, device, dtype,
        )

    mask = _build_border_mask(cur_imsz, border_mask, device, dtype)
    tv_spacing = tuple(cur_pr[d] * cur_gs[d] for d in range(nd))


    # Same loss dtype, objective normalisation, tolerances and warp-grid
    # precision as the one-directional solver (_optimize_level).
    _cdt, _normalize, _tol_grad, _tol_change = _loss_settings(opts, dtype)
    _oscale = _obj_scale(_normalize, cur_imsz, cur_pr)
    _wgdt = _resolve_loss_dtype(opts.get("warp_compute_dtype", torch.float64))
    _icdt = _cdt if _cdt is not None else dtype

    def _forward_losses(knots_fm_, knots_mf_):

        kp_fm = params_phys_to_pix(knots_fm_, cur_pr, spline_order)
        kp_mf = params_phys_to_pix(knots_mf_, cur_pr, spline_order)
        d_fm = disp_from_knots(kp_fm, cur_imsz, cur_gs, spline_order)
        d_mf = disp_from_knots(kp_mf, cur_imsz, cur_gs, spline_order)


        warped_M = warp_image(vol_mov, d_fm, interp_type, grid_dtype=_wgdt)
        warped_F = warp_image(vol_fix, d_mf, interp_type, grid_dtype=_wgdt)
        data_fwd = _build_data_loss(warped_M, vol_fix, opts, cur_pr, mask,
                                    compute_dtype=_cdt)
        data_bwd = _build_data_loss(warped_F, vol_mov, opts, cur_pr, mask,
                                    compute_dtype=_cdt)


        reg_fwd = _per_direction_reg(knots_fm_, opts, tv_spacing, csqrt, isoTV)
        reg_bwd = _per_direction_reg(knots_mf_, opts, tv_spacing, csqrt, isoTV)
        # Dense Jacobian penalty needs the voxel-level field, so it is added
        # here rather than in _per_direction_reg (it used to be ignored
        # whenever ic_weight > 0).
        reg_fwd = reg_fwd + _dense_jac_penalty(d_fm, opts)
        reg_bwd = reg_bwd + _dense_jac_penalty(d_mf, opts)


        ic = inverse_consistency_loss(d_fm.to(_icdt), d_mf.to(_icdt),
                                      cur_pr, mask=mask)

        total = (data_fwd + data_bwd + reg_fwd + reg_bwd
                 + ic_weight * ic) * _oscale
        return total, {
            "data_fwd": data_fwd, "data_bwd": data_bwd,
            "reg_fwd": reg_fwd, "reg_bwd": reg_bwd,
            "ic": ic,
        }

    iter_count = [0]
    last = [None]


    if opt_method == "lbfgs":


        n_knot_elements = knots_fm.numel() + knots_mf.numel()
        if n_knot_elements > 100**3:
            hist_size = 20
        elif n_knot_elements > 50**3:
            hist_size = 30
        else:
            hist_size = 85

        if verbose:
            print(f"  L-BFGS (symmetric): {n_knot_elements} knot elements "
                  f"(both directions), history_size={hist_size}, "
                  f"ic_weight={ic_weight}")

        optimizer = torch.optim.LBFGS(
            [knots_fm, knots_mf],
            max_iter=max_iters,
            history_size=hist_size,
            line_search_fn="strong_wolfe",
            lr=1.0,
            tolerance_grad=_tol_grad * _oscale,
            tolerance_change=_tol_change * _oscale,
        )

        def closure():
            optimizer.zero_grad()
            total, comp = _forward_losses(knots_fm, knots_mf)
            total.backward()
            iter_count[0] += 1
            last[0] = (total.item(), {k: v.item() for k, v in comp.items()})
            if verbose and (iter_count[0] % 20 == 0):
                c = last[0][1]
                print(f"  iter {iter_count[0]:4d}: total={last[0][0]:.4f}  "
                      f"data=({c['data_fwd']:.3f},{c['data_bwd']:.3f})  "
                      f"reg=({c['reg_fwd']:.3f},{c['reg_bwd']:.3f})  "
                      f"ic={c['ic']:.3e}")
            return total

        optimizer.step(closure)
        n_iters = iter_count[0]
        best_fm = knots_fm.detach().clone()
        best_mf = knots_mf.detach().clone()


    elif opt_method == "adam":
        optimizer = torch.optim.Adam([knots_fm, knots_mf], lr=0.05)
        best_total = float("inf")
        best_fm = knots_fm.detach().clone()
        best_mf = knots_mf.detach().clone()

        for it in range(max_iters):
            optimizer.zero_grad()
            total, comp = _forward_losses(knots_fm, knots_mf)
            total.backward()
            optimizer.step()

            tv = total.item()
            last[0] = (tv, {k: v.item() for k, v in comp.items()})
            if tv < best_total:
                best_total = tv
                best_fm = knots_fm.detach().clone()
                best_mf = knots_mf.detach().clone()
            if verbose and (it % 20 == 0 or it == max_iters - 1):
                c = last[0][1]
                print(f"  iter {it:4d}: total={tv:.4f}  "
                      f"data=({c['data_fwd']:.3f},{c['data_bwd']:.3f})  "
                      f"reg=({c['reg_fwd']:.3f},{c['reg_bwd']:.3f})  "
                      f"ic={c['ic']:.3e}")

        n_iters = max_iters

    else:
        raise ValueError(f"Unknown opt_method='{opt_method}' (symmetric path)")


    knots_fm = best_fm.clone().requires_grad_(True)
    knots_mf = best_mf.clone().requires_grad_(True)

    with torch.no_grad():
        total, comp = _forward_losses(knots_fm, knots_mf)

    info = LevelInfo(
        data_loss=float(comp["data_fwd"]),
        data_loss_inv=float(comp["data_bwd"]),
        reg_loss=float(comp["reg_fwd"]),
        reg_loss_inv=float(comp["reg_bwd"]),
        ic_loss=float(comp["ic"]),
        total_loss=total.item() / _oscale,
        total_loss_norm=total.item(),
        oscale=_oscale,
        n_iters=n_iters,
    )

    if verbose:
        print(f"  symmetric done: total={info.total_loss:.4f}  "
              f"data=({info.data_loss:.3f},{info.data_loss_inv:.3f})  "
              f"reg=({info.reg_loss:.3f},{info.reg_loss_inv:.3f})  "
              f"ic={info.ic_loss:.3e}  ({n_iters} fevals)")

    return knots_fm, knots_mf, info


def register_pair_symmetric(
    mov: torch.Tensor,
    fix: torch.Tensor,
    opts: Dict,
):

    device_name = opts.get("device", "cpu")
    dtype = opts.get("dtype", torch.float32)
    device = torch.device(device_name)

    mov = mov.to(device=device, dtype=dtype)
    fix = fix.to(device=device, dtype=dtype)

    nd = mov.dim() - 2
    spatial_size = tuple(mov.shape[2:])

    grid_spacing = tuple(opts.get("grid_spacing", (4,) * nd))[:nd]
    pix_resolution = tuple(opts.get("pix_resolution", (1.0,) * nd))[:nd]
    spline_order = int(opts.get("spline_order", 1))
    k_down = float(opts.get("k_down", 0.7))
    max_iters = int(opts.get("max_iters", 100))
    isoTV = float(opts.get("isoTV", 0.0))
    csqrt = float(opts.get("csqrt", 5e-3))
    border_mask = int(opts.get("border_mask", 3))
    interp_type = int(opts.get("interp_type", 0))
    verbose = bool(opts.get("verbose", False))
    opt_method = str(opts.get("opt_method", "lbfgs")).lower()
    cp_refinements = int(opts.get("cp_refinements", 0))
    ic_weight = float(opts.get("ic_weight", 1.0))


    metric = str(opts.get("metric", "ssd"))
    if metric in ("loc_cc_fftn", "loc_cc_fftn_gpu", "loc_cc_fftn_single",
                  "loc_cc_fftn_gpu_single"):
        metric = "lcc"
    opts["metric"] = metric


    nlvl_raw = opts.get("nlvl", None)
    if nlvl_raw is None or (hasattr(nlvl_raw, '__len__') and len(nlvl_raw) == 0):
        nlvl = compute_nlvl(spatial_size, grid_spacing, k_down)
    else:
        nlvl = int(nlvl_raw)
    nlvl = max(1, nlvl)

    if verbose:
        total_levels = nlvl + cp_refinements
        print(f"Symmetric registration: {nlvl} pyramid levels + "
              f"{cp_refinements} cp_refinements = {total_levels} total, "
              f"opt_method={opt_method}, ic_weight={ic_weight}")


    sigma_blur = 0.4 * (0.5 / k_down)
    interp_mode = "bilinear" if nd == 2 else "trilinear"

    mov_pyr = [mov]
    fix_pyr = [fix]
    pix_res_pyr = [pix_resolution]

    cur_mov = mov
    cur_fix = fix
    cur_pr = pix_resolution

    for lvl in range(1, nlvl):
        cur_mov = _gaussian_blur(cur_mov, sigma_blur, nd)
        cur_fix = _gaussian_blur(cur_fix, sigma_blur, nd)
        new_size = tuple(math.ceil(k_down * s) for s in cur_mov.shape[2:])
        cur_mov = F.interpolate(cur_mov, size=new_size, mode=interp_mode, align_corners=True)
        cur_fix = F.interpolate(cur_fix, size=new_size, mode=interp_mode, align_corners=True)
        cur_pr = tuple(p / k_down for p in cur_pr)
        mov_pyr.append(cur_mov)
        fix_pyr.append(cur_fix)
        pix_res_pyr.append(cur_pr)


    knots_fm = None
    knots_mf = None
    prev_imsz = None
    prev_gs = None
    cur_gs = grid_spacing
    infos = []

    total_levels = nlvl + cp_refinements

    for lvl_idx in range(nlvl - 1, -1, -1):
        vol_mov = mov_pyr[lvl_idx]
        vol_fix = fix_pyr[lvl_idx]
        cur_imsz = tuple(vol_fix.shape[2:])
        cur_pr = pix_res_pyr[lvl_idx]
        cur_gs = grid_spacing

        level_num = nlvl - lvl_idx
        if verbose:
            print(f"Level {level_num}/{total_levels}: "
                  f"size={cur_imsz}, grid={cur_gs}, "
                  f"pix_res={tuple(f'{p:.3f}' for p in cur_pr)}"
                  + (f", lcc_sigma_vox={tuple(f'{v:.2f}' for v in _lcc_sigma_pix(opts, cur_pr, nd))}"
                     if opts.get("metric") == "lcc" else ""))

        if knots_fm is None:
            knots_fm = init_knots(cur_imsz, cur_gs, nd, 1, spline_order,
                                  device=str(device), dtype=dtype)
            knots_mf = init_knots(cur_imsz, cur_gs, nd, 1, spline_order,
                                  device=str(device), dtype=dtype)
        else:
            knots_fm = upsample_knots(
                knots_fm, prev_imsz, prev_gs, cur_imsz, cur_gs,
                spline_order=spline_order, method="direct",
            ).detach().clone().to(device=device, dtype=dtype).requires_grad_(True)
            knots_mf = upsample_knots(
                knots_mf, prev_imsz, prev_gs, cur_imsz, cur_gs,
                spline_order=spline_order, method="direct",
            ).detach().clone().to(device=device, dtype=dtype).requires_grad_(True)

        knots_fm, knots_mf, info = _optimize_level_symmetric(
            knots_fm, knots_mf, vol_mov, vol_fix, opts,
            cur_imsz, cur_pr, cur_gs, nd,
            isoTV, csqrt, border_mask, interp_type, spline_order,
            max_iters, opt_method, ic_weight, verbose, device, dtype,
        )
        info.knots_fm = knots_fm.detach().cpu().clone()
        info.knots_mf = knots_mf.detach().cpu().clone()
        info.grid_spacing = tuple(cur_gs)
        info.image_size = tuple(cur_imsz)
        infos.append(info)

        prev_imsz = cur_imsz
        prev_gs = cur_gs


    for ref_idx in range(cp_refinements):
        vol_mov = mov_pyr[0]
        vol_fix = fix_pyr[0]
        cur_imsz = tuple(vol_fix.shape[2:])
        cur_pr = pix_res_pyr[0]

        old_gs = cur_gs
        cur_gs = tuple(round(g / 2) for g in cur_gs)

        level_num = nlvl + ref_idx + 1
        if verbose:
            print(f"Level {level_num}/{total_levels} (cp_refine {ref_idx+1}): "
                  f"size={cur_imsz}, grid={old_gs}->{cur_gs}, "
                  f"pix_res={tuple(f'{p:.3f}' for p in cur_pr)}")

        knots_fm = upsample_knots(
            knots_fm, prev_imsz, old_gs, cur_imsz, cur_gs,
            spline_order=spline_order, method="direct",
        ).detach().clone().to(device=device, dtype=dtype).requires_grad_(True)
        knots_mf = upsample_knots(
            knots_mf, prev_imsz, old_gs, cur_imsz, cur_gs,
            spline_order=spline_order, method="direct",
        ).detach().clone().to(device=device, dtype=dtype).requires_grad_(True)

        knots_fm, knots_mf, info = _optimize_level_symmetric(
            knots_fm, knots_mf, vol_mov, vol_fix, opts,
            cur_imsz, cur_pr, cur_gs, nd,
            isoTV, csqrt, border_mask, interp_type, spline_order,
            max_iters, opt_method, ic_weight, verbose, device, dtype,
        )
        info.knots_fm = knots_fm.detach().cpu().clone()
        info.knots_mf = knots_mf.detach().cpu().clone()
        info.grid_spacing = tuple(cur_gs)
        info.image_size = tuple(cur_imsz)
        infos.append(info)

        prev_imsz = cur_imsz
        prev_gs = cur_gs


    with torch.no_grad():
        kp_fm = params_phys_to_pix(knots_fm, pix_resolution, spline_order)
        kp_mf = params_phys_to_pix(knots_mf, pix_resolution, spline_order)
        disp_fm_pix = disp_from_knots(kp_fm, spatial_size, cur_gs, spline_order)
        disp_mf_pix = disp_from_knots(kp_mf, spatial_size, cur_gs, spline_order)

        voldef_FM = warp_image(mov, disp_fm_pix, interp_type)
        voldef_MF = warp_image(fix, disp_mf_pix, interp_type)

        pr_t = torch.tensor(pix_resolution, device=device, dtype=dtype)
        pr_t = pr_t.view(1, nd, *([1] * nd))
        Tmm_FM = disp_fm_pix * pr_t
        Tmm_MF = disp_mf_pix * pr_t

    return (
        voldef_FM, voldef_MF,
        Tmm_FM, Tmm_MF,
        knots_fm.detach(), knots_mf.detach(),
        infos,
    )


def _optimize_level_admm_symmetric(
    knots_fm, knots_mf, vol_mov, vol_fix, opts,
    cur_imsz, cur_pr, cur_gs, nd,
    isoTV, csqrt, border_mask_px, interp_type, spline_order,
    max_iters, ic_weight, verbose, device, dtype,
):
    mask = _build_border_mask(cur_imsz, border_mask_px, device, dtype)
    tv_spacing = tuple(cur_pr[d] * cur_gs[d] for d in range(nd))
    _cdt, _normalize, _tol_grad, _tol_change = _loss_settings(opts, dtype)
    _oscale = _obj_scale(_normalize, cur_imsz, cur_pr)
    _wgdt = _resolve_loss_dtype(opts.get("warp_compute_dtype", torch.float64))
    _icdt = _cdt if _cdt is not None else dtype

    lam = isoTV
    eta = 1.0
    for s in tv_spacing[:nd]:
        eta *= s


    admm_max_outer = int(opts.get("admm_max_outer", 10))
    lbfgs_iters = int(opts.get("admm_lbfgs_iters", 30))
    mu = float(opts.get("admm_mu", 10.0))
    tau = float(opts.get("admm_tau", 1.3))
    tol = float(opts.get("admm_tol", 1e-2))


    single_rho = bool(opts.get("admm_single_rho", True))

    rho_user = opts.get("admm_rho", None)
    if rho_user is not None and float(rho_user) > 0:
        rho_fm = float(rho_user)
        rho_mf = float(rho_user)
    else:
        rho_fm = max(lam * eta, 1.0)
        rho_mf = max(lam * eta, 1.0)

    if verbose:
        print(f"    ADMM-sym: rho0={rho_fm:.2f} (lam*eta={lam*eta:.2f}), "
              f"mu={mu}, tau={tau}, ic_weight={ic_weight}, "
              f"single_rho={single_rho}, "
              f"outer={admm_max_outer}, inner_lbfgs={lbfgs_iters}")


    with torch.no_grad():
        Dk_fm = _compute_D_knots(knots_fm, tv_spacing, nd, differentiable=False)
        Dk_mf = _compute_D_knots(knots_mf, tv_spacing, nd, differentiable=False)
    Z_fm, U_fm = Dk_fm.clone(), torch.zeros_like(Dk_fm)
    Z_mf, U_mf = Dk_mf.clone(), torch.zeros_like(Dk_mf)


    knots_fm_opt = knots_fm.detach().clone().requires_grad_(True)
    knots_mf_opt = knots_mf.detach().clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [knots_fm_opt, knots_mf_opt],
        max_iter=lbfgs_iters,
        history_size=5,
        line_search_fn="strong_wolfe",
        lr=1.0,
        tolerance_grad=_tol_grad * _oscale,
        tolerance_change=_tol_change * _oscale,
    )

    total_func_evals = 0
    knots_fm_prev = knots_fm_opt.detach().clone()
    knots_mf_prev = knots_mf_opt.detach().clone()


    admm_hist = {
        'single_rho': single_rho,
        'data_fwd': [], 'data_bwd': [],
        'penalty_fm': [], 'penalty_mf': [],
        'tv_fm': [], 'tv_mf': [],
        'ic': [], 'total': [],
        'rho_fm': [], 'rho_mf': [],
        'r_fm': [], 's_fm': [], 'r_mf': [], 's_mf': [],
        'dk_fm': [], 'dk_mf': [],
        'z_sparse_fm': [], 'z_sparse_mf': [],
    }

    for admm_it in range(admm_max_outer):


        target_fm = (Z_fm - U_fm).detach()
        target_mf = (Z_mf - U_mf).detach()
        cur_rho_fm = rho_fm
        cur_rho_mf = rho_mf

        func_evals = [0]

        def closure():
            optimizer.zero_grad()

            kp_fm = params_phys_to_pix(knots_fm_opt, cur_pr, spline_order)
            kp_mf = params_phys_to_pix(knots_mf_opt, cur_pr, spline_order)
            d_fm = disp_from_knots(kp_fm, cur_imsz, cur_gs, spline_order)
            d_mf = disp_from_knots(kp_mf, cur_imsz, cur_gs, spline_order)

            warped_M = warp_image(vol_mov, d_fm, interp_type, grid_dtype=_wgdt)
            warped_F = warp_image(vol_fix, d_mf, interp_type, grid_dtype=_wgdt)
            data_fwd = _build_data_loss(warped_M, vol_fix, opts, cur_pr, mask,
                                        compute_dtype=_cdt)
            data_bwd = _build_data_loss(warped_F, vol_mov, opts, cur_pr, mask,
                                        compute_dtype=_cdt)

            Dk_fm_cur = _compute_D_knots(
                knots_fm_opt, tv_spacing, nd, differentiable=True)
            Dk_mf_cur = _compute_D_knots(
                knots_mf_opt, tv_spacing, nd, differentiable=True)
            penalty_fm = (cur_rho_fm / 2.0) * ((Dk_fm_cur - target_fm) ** 2).sum()
            penalty_mf = (cur_rho_mf / 2.0) * ((Dk_mf_cur - target_mf) ** 2).sum()
            penalty_fm = penalty_fm + _dense_jac_penalty(d_fm, opts)
            penalty_mf = penalty_mf + _dense_jac_penalty(d_mf, opts)

            total = data_fwd + data_bwd + penalty_fm + penalty_mf
            if ic_weight > 0:
                total = total + ic_weight * inverse_consistency_loss(
                    d_fm.to(_icdt), d_mf.to(_icdt), cur_pr, mask=mask)
            total = total * _oscale

            total.backward()
            func_evals[0] += 1
            return total

        optimizer.step(closure)
        total_func_evals += func_evals[0]


        with torch.no_grad():
            Dk_fm = _compute_D_knots(knots_fm_opt, tv_spacing, nd, differentiable=False)
            Dk_mf = _compute_D_knots(knots_mf_opt, tv_spacing, nd, differentiable=False)


        Z_fm_old, Z_mf_old = Z_fm.clone(), Z_mf.clone()
        Z_fm = _block_soft_threshold(Dk_fm + U_fm, lam * eta / rho_fm)
        Z_mf = _block_soft_threshold(Dk_mf + U_mf, lam * eta / rho_mf)


        U_fm = U_fm + Dk_fm - Z_fm
        U_mf = U_mf + Dk_mf - Z_mf


        r_fm = Dk_fm - Z_fm
        r_mf = Dk_mf - Z_mf
        with torch.no_grad():
            s_fm = rho_fm * _D_adjoint(Z_fm - Z_fm_old, nd, tv_spacing)
            s_mf = rho_mf * _D_adjoint(Z_mf - Z_mf_old, nd, tv_spacing)
        r_fm_norm, s_fm_norm = r_fm.norm().item(), s_fm.norm().item()
        r_mf_norm, s_mf_norm = r_mf.norm().item(), s_mf.norm().item()

        rho_fm_old = rho_fm
        rho_mf_old = rho_mf

        if single_rho:


            r_comb = (r_fm_norm ** 2 + r_mf_norm ** 2) ** 0.5
            s_comb = (s_fm_norm ** 2 + s_mf_norm ** 2) ** 0.5
            rho_shared = rho_fm
            if r_comb >= mu * s_comb:
                rho_shared = tau * rho_shared
                U_fm = U_fm / tau
                U_mf = U_mf / tau
            elif s_comb >= mu * r_comb:
                rho_shared = rho_shared / tau
                U_fm = U_fm * tau
                U_mf = U_mf * tau
            rho_fm = rho_shared
            rho_mf = rho_shared
        else:


            if r_fm_norm >= mu * s_fm_norm:
                rho_fm = tau * rho_fm
                U_fm = U_fm / tau
            elif s_fm_norm >= mu * r_fm_norm:
                rho_fm = rho_fm / tau
                U_fm = U_fm * tau

            if r_mf_norm >= mu * s_mf_norm:
                rho_mf = tau * rho_mf
                U_mf = U_mf / tau
            elif s_mf_norm >= mu * r_mf_norm:
                rho_mf = rho_mf / tau
                U_mf = U_mf * tau


        with torch.no_grad():
            dk_fm = (knots_fm_opt - knots_fm_prev).abs().max().item()
            dk_mf = (knots_mf_opt - knots_mf_prev).abs().max().item()
            knots_fm_prev = knots_fm_opt.detach().clone()
            knots_mf_prev = knots_mf_opt.detach().clone()


        with torch.no_grad():
            kp_m_fm = params_phys_to_pix(knots_fm_opt, cur_pr, spline_order)
            kp_m_mf = params_phys_to_pix(knots_mf_opt, cur_pr, spline_order)
            d_m_fm = disp_from_knots(kp_m_fm, cur_imsz, cur_gs, spline_order)
            d_m_mf = disp_from_knots(kp_m_mf, cur_imsz, cur_gs, spline_order)
            wM = warp_image(vol_mov, d_m_fm, interp_type, grid_dtype=_wgdt)
            wF = warp_image(vol_fix, d_m_mf, interp_type, grid_dtype=_wgdt)
            data_fwd_m = _build_data_loss(wM, vol_fix, opts, cur_pr, mask,
                                          compute_dtype=_cdt).item()
            data_bwd_m = _build_data_loss(wF, vol_mov, opts, cur_pr, mask,
                                          compute_dtype=_cdt).item()

            penalty_fm_m = (rho_fm / 2.0) * (r_fm ** 2).sum().item()
            penalty_mf_m = (rho_mf / 2.0) * (r_mf ** 2).sum().item()

            z_norms_fm = torch.sqrt((Z_fm ** 2).sum(dim=1))
            z_norms_mf = torch.sqrt((Z_mf ** 2).sum(dim=1))
            tv_fm_m = (lam * eta * z_norms_fm.sum() / knots_fm_opt.shape[0]).item()
            tv_mf_m = (lam * eta * z_norms_mf.sum() / knots_mf_opt.shape[0]).item()
            z_sparse_fm = (z_norms_fm < 1e-10).float().mean().item()
            z_sparse_mf = (z_norms_mf < 1e-10).float().mean().item()

            ic_m = inverse_consistency_loss(d_m_fm.to(_icdt), d_m_mf.to(_icdt),
                                            cur_pr, mask=mask).item()
            jac_m = (float(_dense_jac_penalty(d_m_fm, opts))
                     + float(_dense_jac_penalty(d_m_mf, opts)))
            total_m = (data_fwd_m + data_bwd_m + tv_fm_m + tv_mf_m + jac_m
                       + ic_weight * ic_m)

        admm_hist['data_fwd'].append(data_fwd_m)
        admm_hist['data_bwd'].append(data_bwd_m)
        admm_hist['penalty_fm'].append(penalty_fm_m)
        admm_hist['penalty_mf'].append(penalty_mf_m)
        admm_hist['tv_fm'].append(tv_fm_m)
        admm_hist['tv_mf'].append(tv_mf_m)
        admm_hist['ic'].append(ic_m)
        admm_hist['total'].append(total_m)
        admm_hist['rho_fm'].append(rho_fm)
        admm_hist['rho_mf'].append(rho_mf)
        admm_hist['r_fm'].append(r_fm_norm)
        admm_hist['s_fm'].append(s_fm_norm)
        admm_hist['r_mf'].append(r_mf_norm)
        admm_hist['s_mf'].append(s_mf_norm)
        admm_hist['dk_fm'].append(dk_fm)
        admm_hist['dk_mf'].append(dk_mf)
        admm_hist['z_sparse_fm'].append(z_sparse_fm)
        admm_hist['z_sparse_mf'].append(z_sparse_mf)

        if verbose:
            print(f"    ADMM-sym {admm_it+1:3d}: "
                  f"data=({data_fwd_m:.2f},{data_bwd_m:.2f}) "
                  f"tv=({tv_fm_m:.2f},{tv_mf_m:.2f}) ic={ic_m:.3e} "
                  f"rho=({rho_fm:.2f},{rho_mf:.2f}) "
                  f"r=({r_fm_norm:.2e},{r_mf_norm:.2e}) "
                  f"dk=({dk_fm:.2e},{dk_mf:.2e}) "
                  f"fevals={func_evals[0]}")

        if max(dk_fm, dk_mf) < tol and admm_it > 0:
            if verbose:
                print(f"    ADMM-sym converged at iter {admm_it+1}")
            break


    with torch.no_grad():
        kp_fm = params_phys_to_pix(knots_fm_opt, cur_pr, spline_order)
        kp_mf = params_phys_to_pix(knots_mf_opt, cur_pr, spline_order)
        d_fm = disp_from_knots(kp_fm, cur_imsz, cur_gs, spline_order)
        d_mf = disp_from_knots(kp_mf, cur_imsz, cur_gs, spline_order)
        wM = warp_image(vol_mov, d_fm, interp_type, grid_dtype=_wgdt)
        wF = warp_image(vol_fix, d_mf, interp_type, grid_dtype=_wgdt)
        final_data_fwd = _build_data_loss(wM, vol_fix, opts, cur_pr, mask,
                                          compute_dtype=_cdt).item()
        final_data_bwd = _build_data_loss(wF, vol_mov, opts, cur_pr, mask,
                                          compute_dtype=_cdt).item()

        z_norms_fm = torch.sqrt((Z_fm ** 2).sum(dim=1))
        z_norms_mf = torch.sqrt((Z_mf ** 2).sum(dim=1))
        final_reg_fwd = (lam * eta * z_norms_fm.sum() / knots_fm_opt.shape[0]).item()
        final_reg_bwd = (lam * eta * z_norms_mf.sum() / knots_mf_opt.shape[0]).item()
        final_reg_fwd += float(_dense_jac_penalty(d_fm, opts))
        final_reg_bwd += float(_dense_jac_penalty(d_mf, opts))
        final_ic = inverse_consistency_loss(d_fm.to(_icdt), d_mf.to(_icdt),
                                            cur_pr, mask=mask).item()

    knots_fm_out = knots_fm_opt.detach().clone().requires_grad_(True)
    knots_mf_out = knots_mf_opt.detach().clone().requires_grad_(True)

    info = LevelInfo(
        data_loss=final_data_fwd,
        data_loss_inv=final_data_bwd,
        reg_loss=final_reg_fwd,
        reg_loss_inv=final_reg_bwd,
        ic_loss=final_ic,
        total_loss=(final_data_fwd + final_data_bwd
                    + final_reg_fwd + final_reg_bwd
                    + ic_weight * final_ic),
        total_loss_norm=(final_data_fwd + final_data_bwd
                         + final_reg_fwd + final_reg_bwd
                         + ic_weight * final_ic) * _oscale,
        oscale=_oscale,
        n_iters=total_func_evals,
        admm_history=admm_hist,
    )

    if verbose:
        print(f"  ADMM-sym done: data=({final_data_fwd:.2f},{final_data_bwd:.2f}) "
              f"reg=({final_reg_fwd:.2f},{final_reg_bwd:.2f}) "
              f"ic={final_ic:.3e} ({total_func_evals} fevals)")

    return knots_fm_out, knots_mf_out, info