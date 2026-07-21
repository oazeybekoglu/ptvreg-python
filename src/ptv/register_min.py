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
from .regularizers_jac import jac_reg_regularization
from .upsample_knots import upsample_knots
from .create_pyramid import compute_nlvl


@dataclass
class LevelInfo:
    data_loss: float = 0.0
    reg_loss: float = 0.0
    total_loss: float = 0.0
    n_iters: int = 0
    admm_history: dict = None
    knots_fm: object = None
    grid_spacing: tuple = None
    image_size: tuple = None


def _build_data_loss(warped, fixed, opts, pix_res_cur, mask):
    metric = opts["metric"]
    nd = warped.dim() - 2

    if metric == "ssd":
        return ssd_loss(warped, fixed, pix_res_cur, mask)

    elif metric == "sad":
        return sad_loss(warped, fixed, pix_res_cur, mask)

    elif metric == "lcc":
        mp = opts.get("metric_param", (7.0,) * nd)
        if opts.get("scale_metric_param", True):
            sigma_pix = tuple(mp[d] / pix_res_cur[d] for d in range(nd))
        else:
            pr0 = opts.get("pix_resolution", (1.0,) * nd)
            sigma_pix = tuple(mp[d] / pr0[d] for d in range(nd))

        eps = opts.get("deps", 1e-4)
        use_abs = opts.get("loc_cc_abs", False)
        return lcc_loss(warped, fixed, pix_res_cur, sigma_pix,
                        mask=mask, eps=eps, use_abs=use_abs)

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
            tolerance_grad=1e-8,
            tolerance_change=1e-9,
        )

        iter_count = [0]
        last_loss = [float("inf")]

        def closure():
            optimizer.zero_grad()
            kp = params_phys_to_pix(knots, cur_pr, spline_order)
            d = disp_from_knots(kp, cur_imsz, cur_gs, spline_order)
            w = warp_image(vol_mov, d, interp_type)
            data_loss = _build_data_loss(w, vol_fix, opts, cur_pr, mask)

            reg_loss = torch.tensor(0.0, device=device, dtype=dtype)
            if isoTV > 0:
                reg_loss = reg_loss + isoTV * isotropic_tv_regularization(
                    knots, tv_spacing, csqrt)
            _jac_reg = float(opts.get("jac_reg", 0.0))
            if _jac_reg > 0:
                reg_loss = reg_loss + _jac_reg * jac_reg_regularization(knots, tv_spacing)
            total = data_loss + reg_loss
            total.backward()

            last_loss[0] = total.item()
            iter_count[0] += 1
            if verbose and (iter_count[0] % 20 == 0):
                print(f"  iter {iter_count[0]:4d}: loss = {total.item():.6f}")

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
            w = warp_image(vol_mov, d, interp_type)
            data_loss = _build_data_loss(w, vol_fix, opts, cur_pr, mask)

            reg_loss = torch.tensor(0.0, device=device, dtype=dtype)
            if isoTV > 0:
                reg_loss = reg_loss + isoTV * isotropic_tv_regularization(
                    knots, tv_spacing, csqrt)
            _jac_reg = float(opts.get("jac_reg", 0.0))
            if _jac_reg > 0:
                reg_loss = reg_loss + _jac_reg * jac_reg_regularization(knots, tv_spacing)
            total = data_loss + reg_loss
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
        w = warp_image(vol_mov, d, interp_type)
        final_data = _build_data_loss(w, vol_fix, opts, cur_pr, mask).item()
        final_reg = 0.0
        if isoTV > 0:
            final_reg += (isoTV * isotropic_tv_regularization(
                knots, tv_spacing, csqrt)).item()
        if float(opts.get("jac_reg", 0)) > 0:
            final_reg += (float(opts.get("jac_reg", 0)) * jac_reg_regularization(knots, tv_spacing)).item()

    info = LevelInfo(
        data_loss=final_data,
        reg_loss=final_reg,
        total_loss=final_data + final_reg,
        n_iters=n_iters,
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
        tolerance_grad=1e-8,
        tolerance_change=1e-9,
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
            w = warp_image(vol_mov, d, interp_type)
            data_loss = _build_data_loss(w, vol_fix, opts, cur_pr, mask)


            Dk_cur = _compute_D_knots(
                knots_opt, tv_spacing, nd, differentiable=True
            )
            penalty = (cur_rho / 2.0) * ((Dk_cur - target) ** 2).sum()


            _jac_reg = float(opts.get("jac_reg", 0.0))
            if _jac_reg > 0:
                penalty = penalty + _jac_reg * jac_reg_regularization(
                    knots_opt, tv_spacing)

            total = data_loss + penalty
            total.backward()

            func_evals[0] += 1
            return total

        optimizer.step(closure)
        total_func_evals += func_evals[0]


        if verbose:
            with torch.no_grad():
                kp_k = params_phys_to_pix(knots_opt, cur_pr, spline_order)
                d_k = disp_from_knots(kp_k, cur_imsz, cur_gs, spline_order)
                w_k = warp_image(vol_mov, d_k, interp_type)
                kdata = _build_data_loss(w_k, vol_fix, opts, cur_pr, mask).item()
                Dk_tmp = _compute_D_knots(knots_opt, tv_spacing, nd, differentiable=False)
                kpenalty = (cur_rho / 2.0) * ((Dk_tmp - target) ** 2).sum().item()
            print(f"      k-update: data={kdata:.2f}  penalty={kpenalty:.2f}  "
                  f"total={kdata+kpenalty:.2f}  ({func_evals[0]} fevals)")


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

        r_norm = r.norm().item()
        s_norm = s.norm().item()

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
            w_mon = warp_image(vol_mov, d_mon, interp_type)
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

        if verbose:
            print(f"    ADMM iter {admm_it+1:3d}: "
                  f"data={data_loss_mon:.2f}  tv={tv_val:.2f}  total={total_mon:.2f}  "
                  f"r={r_norm:.4e}  s={s_norm:.4e}  "
                  f"rho={rho:.2f}  thr={cur_threshold:.4f}  "
                  f"dk={k_change:.4e}  Z_sparse={z_zero_frac:.1%}  "
                  f"fevals={func_evals[0]}")

        if k_change < tol and admm_it > 0:
            if verbose:
                print(f"    ADMM converged at iter {admm_it+1} "
                      f"(dk={k_change:.2e} < tol={tol:.2e})")
            break


    with torch.no_grad():
        kp = params_phys_to_pix(knots_opt, cur_pr, spline_order)
        d = disp_from_knots(kp, cur_imsz, cur_gs, spline_order)
        w = warp_image(vol_mov, d, interp_type)
        final_data = _build_data_loss(w, vol_fix, opts, cur_pr, mask).item()


        z_norms = torch.sqrt((Z ** 2).sum(dim=1))
        final_reg = (lam * eta * z_norms.sum() / knots_opt.shape[0]).item()
        _jac_reg = float(opts.get("jac_reg", 0.0))
        if _jac_reg > 0:
            final_reg += (_jac_reg * jac_reg_regularization(
                knots_opt, tv_spacing)).item()

    knots_out = knots_opt.detach().clone().requires_grad_(True)

    info = LevelInfo(
        data_loss=final_data,
        reg_loss=final_reg,
        total_loss=final_data + final_reg,
        n_iters=total_func_evals,
        admm_history=admm_hist,
    )

    if verbose:
        print(f"  ADMM done: data={final_data:.4f} reg={final_reg:.4f} "
              f"total={final_data+final_reg:.4f} ({total_func_evals} fevals)")

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

    device_name = opts.get("device", "cpu")
    dtype = opts.get("dtype", torch.float64)
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
                  f"pix_res={tuple(f'{p:.3f}' for p in cur_pr)}")

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
