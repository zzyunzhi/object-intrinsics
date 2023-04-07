import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
try:
    import mcubes
    from icecream import ic
except:
    pass

logger = logging.getLogger(__name__)


def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution, device='cuda').split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution, device='cuda').split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution, device='cuda').split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    # print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples, device='cuda')
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device='cuda')

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


class NeuSRenderer:
    def __init__(self,
                 nerf,
                 sdf_network,
                 deviation_network,
                 color_network,
                 n_samples,
                 n_importance,
                 n_outside,
                 up_sample_steps,
                 perturb):
        self.nerf = nerf
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.color_network = color_network
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.n_outside = n_outside
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb

    def render_core_outside(self, rays_o, rays_d, z_vals, sample_dist, nerf, background_rgb=None):
        """
        Render background
        """
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.tensor([sample_dist], device='cuda', dtype=torch.float32).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # batch_size, n_samples, 3

        dis_to_center = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).clip(1.0, 1e10)
        pts = torch.cat([pts / dis_to_center, 1.0 / dis_to_center], dim=-1)       # batch_size, n_samples, 4

        dirs = rays_d[:, None, :].expand(batch_size, n_samples, 3)

        pts = pts.reshape(-1, 3 + int(self.n_outside > 0))
        dirs = dirs.reshape(-1, 3)

        density, sampled_color = nerf(pts, dirs)
        alpha = 1.0 - torch.exp(-F.softplus(density.reshape(batch_size, n_samples)) * dists)
        alpha = alpha.reshape(batch_size, n_samples)
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1], device='cuda'), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        # sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
        sampled_color = sampled_color.unflatten(0, (batch_size, n_samples))
        color = (weights[:, :, None] * sampled_color).sum(dim=1)
        if background_rgb is not None:
            color = color + background_rgb * (1.0 - weights.sum(dim=-1, keepdim=True))

        return {
            'color': color,
            'sampled_color': sampled_color,
            'alpha': alpha,
            'weights': weights,
        }

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1], device='cuda'), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1], device='cuda'), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False, z=None, w=None):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3), z=z, w=w).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf

    def render_core(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    sample_dist,
                    sdf_network,
                    deviation_network,
                    color_network,
                    siren_network=None,
                    z=None,
                    w=None,
                    background_alpha=None,
                    background_sampled_color=None,
                    background_rgb=None,
                    cos_anneal_ratio=0.0,
                    second_order=None,
                    compute_color=True):
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        if not isinstance(sample_dist, torch.Tensor):
            # default behavior
            dists = torch.cat([dists, torch.tensor([sample_dist], device='cuda', dtype=torch.float32).expand(dists[..., :1].shape)], -1)
        else:
            dists = torch.cat([dists, sample_dist], dim=-1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)

        if siren_network is not None:
            siren_out = siren_network(pts)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        if siren_network is not None:
            siren_out = siren_out.flatten(0, 1)
            sdf_nn_output = sdf_network(pts, siren_out)
        elif z is not None or w is not None:
            sdf_nn_output = sdf_network(pts, z=z, w=w)
        else:
            sdf_nn_output = sdf_network(pts)
        sdf = sdf_nn_output[:, :1]
        feature_vector = sdf_nn_output[:, 1:]

        if siren_network is not None:
            gradients = sdf_network.gradient(pts, siren_out).squeeze()
        elif z is not None or w is not None:
            if second_order is True:
                gradients, hessian = sdf_network.gradient(pts, z=z, w=w, second_order=second_order)
            else:
                gradients = sdf_network.gradient(pts, z=z, w=w)
        else:
            gradients = sdf_network.gradient(pts).squeeze()
        # sampled_color = color_network(pts, gradients, dirs, feature_vector).reshape(batch_size, n_samples, 3)
        if compute_color:
            if siren_network is not None:
                sampled_color = color_network(pts, gradients, dirs, feature_vector, siren_out)
            elif z is not None or w is not None:
                sampled_color = color_network(pts, gradients, dirs, feature_vector, z=z, w=w)
            else:
                sampled_color = color_network(pts, gradients, dirs, feature_vector)
            sampled_color = sampled_color.unflatten(0, (batch_size, n_samples))

        inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)           # Single parameter
        inv_s = inv_s.expand(batch_size * n_samples, 1)

        true_cos = (dirs * gradients).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)

        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        inside_sphere = (pts_norm < 1.0).float().detach()
        relax_inside_sphere = (pts_norm < 1.2).float().detach()

        # Render with background
        if background_alpha is not None:
            alpha = alpha * inside_sphere + background_alpha[:, :n_samples] * (1.0 - inside_sphere)
            alpha = torch.cat([alpha, background_alpha[:, n_samples:]], dim=-1)
            sampled_color = sampled_color * inside_sphere[:, :, None] +\
                            background_sampled_color[:, :n_samples] * (1.0 - inside_sphere)[:, :, None]
            sampled_color = torch.cat([sampled_color, background_sampled_color[:, n_samples:]], dim=1)

        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1], device='cuda'), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        weights_sum = weights.sum(dim=-1, keepdim=True)

        if compute_color:
            color = (sampled_color * weights[:, :, None]).sum(dim=1)
            if background_rgb is not None:    # Fixed background, usually black
                color = color + background_rgb * (1.0 - weights_sum)

        # Eikonal loss
        gradient_error = (torch.linalg.norm(gradients.reshape(batch_size, n_samples, 3), ord=2,
                                            dim=-1) - 1.0) ** 2
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)

        # Divergence
        if False: #divergence is not None:
            divergence = divergence.reshape(batch_size, n_samples, 3, 3)
            if torch.any(divergence.isnan()):
                logger.error(f'encountered nan in divergence: {torch.sum(divergence.isnan()) / divergence.numel()}, '
                             f'{torch.sum(divergence.isinf()) / divergence.numel()}, '
                             f'{torch.min(divergence)}, {torch.max(divergence)}')
            divergence[divergence.isnan()] = 0
            divergence_reg = torch.diagonal(divergence, dim1=-2, dim2=-1).sum(-1)  # (bs, n_pts)
            # # div type L2
            # nonmnfld_divergence_term = torch.clamp(torch.square(divergence_reg), 0.1, 50)
            # div type L1
            # print('divergence', torch.abs(divergence_reg).min(), torch.abs(divergence_reg).max(), divergence_reg.shape)
            divergence_reg = torch.clamp(torch.abs(divergence_reg), 0.1, 50)
            divergence_reg = (relax_inside_sphere * divergence_reg).sum() / (relax_inside_sphere.sum() + 1e-5)

        ret = {
            'sdf': sdf.reshape(batch_size, n_samples),
            'dists': dists,
            'gradients': gradients.reshape(batch_size, n_samples, 3),
            's_val': 1.0 / inv_s,
            'mid_z_vals': mid_z_vals,
            'weights': weights,
            'cdf': c.reshape(batch_size, n_samples),
            'gradient_error': gradient_error,
            'surface_loss': torch.exp(-1e2 * sdf.abs()).mean(),
            'inside_sphere': inside_sphere,
            'pts_norm': pts_norm,
            'pts': pts.reshape(batch_size, n_samples, 3),
            'alpha': alpha.reshape(batch_size, n_samples),
        }
        if compute_color:
            ret['color'] = color
            ret['raw_color'] = sampled_color
        if second_order is True:
            ret['hessian'] = hessian.reshape(batch_size, n_samples, 3, 3)
        return ret

    def render(self, rays_o, rays_d, near, far, perturb_overwrite=-1, background_rgb=None, cos_anneal_ratio=0.0,
               siren_network=None, z=None, w=None, second_order=None, compute_color=True, compute_sample_dist=False, blend_background=False):
        batch_size = len(rays_o)
        if not compute_sample_dist:
            # default training behavior
            sample_dist = 2.0 / self.n_samples   # Assuming the region of interest is a unit sphere
        else:
            sample_dist = (far - near) / self.n_samples
        z_vals = torch.linspace(0.0, 1.0, self.n_samples, device='cuda')
        z_vals = near + (far - near) * z_vals[None, :]

        z_vals_outside = None
        if self.n_outside > 0:
            z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside, device='cuda')

        n_samples = self.n_samples
        perturb = self.perturb

        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1], device='cuda') - 0.5)
            z_vals = z_vals + t_rand * 2.0 / self.n_samples

            if self.n_outside > 0:
                mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
                upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
                lower = torch.cat([z_vals_outside[..., :1], mids], -1)
                t_rand = torch.rand([batch_size, z_vals_outside.shape[-1]], device='cuda')
                z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand

        if self.n_outside > 0:
            z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / self.n_samples

        background_alpha = None
        background_sampled_color = None

        # Up sample
        if self.n_importance > 0:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                if siren_network is not None:
                    siren_out = siren_network(pts)
                    sdf = self.sdf_network.sdf(pts.flatten(0, 1), siren_out.flatten(0, 1)).reshape(batch_size, self.n_samples)
                elif z is not None or w is not None:
                    sdf = self.sdf_network.sdf(pts.flatten(0, 1), z=z, w=w).reshape(batch_size, self.n_samples)
                else:
                    sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, self.n_samples)

                for i in range(self.up_sample_steps):
                    new_z_vals = self.up_sample(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                self.n_importance // self.up_sample_steps,
                                                64 * 2**i)
                    z_vals, sdf = self.cat_z_vals(rays_o,
                                                  rays_d,
                                                  z_vals,
                                                  new_z_vals,
                                                  sdf,
                                                  last=(i + 1 == self.up_sample_steps),
                                                  z=z, w=w)

            n_samples = self.n_samples + self.n_importance

        # Background model
        if self.n_outside > 0:
            z_vals_feed = torch.cat([z_vals, z_vals_outside], dim=-1)
            z_vals_feed, _ = torch.sort(z_vals_feed, dim=-1)
            ret_outside = self.render_core_outside(rays_o, rays_d, z_vals_feed, sample_dist, self.nerf)

            background_sampled_color = ret_outside['sampled_color']
            background_alpha = ret_outside['alpha']

        if not blend_background:
            background_sampled_color = None
            background_alpha = None

        # Render core
        ret_fine = self.render_core(rays_o,
                                    rays_d,
                                    z_vals,
                                    sample_dist,
                                    self.sdf_network,
                                    self.deviation_network,
                                    self.color_network,
                                    siren_network=siren_network,
                                    background_rgb=background_rgb,
                                    background_alpha=background_alpha,
                                    background_sampled_color=background_sampled_color,
                                    cos_anneal_ratio=cos_anneal_ratio,
                                    z=z, w=w, second_order=second_order, compute_color=compute_color)

        weights = ret_fine['weights']
        weights_sum = weights.sum(dim=-1, keepdim=True)

        ret = {
            's_val': ret_fine['s_val'].reshape(batch_size, n_samples).mean(dim=-1, keepdim=True),
            'cdf_fine': ret_fine['cdf'],
            'weight_sum': weights_sum,
            'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
            'gradients': ret_fine['gradients'],
            'weights': weights,
            'gradient_error': ret_fine['gradient_error'],
            'inside_sphere': ret_fine['inside_sphere'],
            # 'z_vals': z_vals,
            'mid_z_vals': ret_fine['mid_z_vals'],
            'surface_loss': ret_fine['surface_loss'],
            'sdf': ret_fine['sdf'],
            'pts_norm': ret_fine['pts_norm'],
            'pts': ret_fine['pts'],
        }
        if compute_color:
            ret['color_fine'] = ret_fine['color']
            ret['raw_color'] = ret_fine['raw_color']
        if second_order:
            ret['hessian'] = ret_fine['hessian']
        if self.n_outside > 0:
            ret['background_sampled_color'] = ret_outside['sampled_color'].reshape(batch_size, n_samples + self.n_outside, 3)
            ret['background_alpha'] = ret_outside['alpha'].reshape(batch_size, n_samples + self.n_outside)
            ret['alpha'] = ret_fine['alpha']
        return ret

    def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.0,
                         siren_network=None, z=None, w=None):
        if siren_network is not None:
            def query_func(pts):
                siren_out = siren_network(pts)
                return -self.sdf_network.sdf(pts, siren_out)
        elif z is not None or w is not None:
            def query_func(pts):
                return -self.sdf_network.sdf(pts, z=z, w=w)
        else:
            def query_func(pts):
                return -self.sdf_network.sdf(pts)

        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=query_func)
