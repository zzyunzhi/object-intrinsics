import numpy as np
import math
from tu.train_setup import count_not_trainable_parameters, count_trainable_parameters, count_parameters
import os
import torch.nn as nn
from ..utils.pose import invert_rot_t
from tu.utils.config import build_from_config
from ..utils.prior import load_bg_color_fn
import torch
from typing import Dict, Union
import logging


MAX_RAY_BATCH_SIZE = 128 * 128 * 1

logger = logging.getLogger(__name__)


class Generator(nn.Module):
    def __init__(self, color_network, sdf_network, deviation_network, light_network, camera,
                 z_dim,
                 resolution,
                 scene_resolution,
                 renderer,
                 anneal_end,
                 pose_prior,
                 ):
        super().__init__()
        self.resolution = resolution
        self.scene_resolution = scene_resolution
        self.z_dim = z_dim
        self.anneal_end = anneal_end

        self.register_buffer('it', torch.tensor(-1, dtype=torch.long))
        self.bg_color = load_bg_color_fn('random', self.resolution, self.resolution)

        """ Build environmental extrinsics networks """

        self.camera = build_from_config(camera)  # camera for the scene, NOT for the object crop
        self.light = build_from_config(light_network)
        self.pose_prior = build_from_config(pose_prior)

        """ Build object intrinsics networks """

        self.color_network = build_from_config(color_network)
        self.sdf_network = build_from_config(sdf_network)
        self.deviation_network = build_from_config(deviation_network)

        """ Build renderer """

        self.renderer = build_from_config(
            renderer,
            nerf=None,
            sdf_network=self.sdf_network,
            deviation_network=self.deviation_network,
            color_network=self.color_network,
        )

        for k, m in self.named_children():
            logger.info(f"generator params {k}, "
                        f"{count_parameters(m, verbose=False)} = "
                        f"{count_trainable_parameters(m, verbose=False)} + "
                        f"{count_not_trainable_parameters(m, verbose=False)}")

    def sample_prior(self, bs: int, data: Dict) -> Dict:
        prior_info = {}
        if 'b2w' in data:
            assert not self.training
            b2w = data['b2w']
        else:
            b2w = torch.tensor(self.pose_prior(bs), dtype=torch.float32, device='cuda')
        w2b = invert_rot_t(b2w)
        c2b = torch.einsum('bij,jk->bik', w2b, self.camera.c2w)  # (b, 4, 4)
        light = self.light.batch_transform(w2b=w2b)
        prior_info['c2b'] = c2b
        prior_info['b2w'] = b2w
        prior_info['light'] = light
        return prior_info

    def render_maps(self, bs, render_out, rays_info, prior_info, return_raw):
        h = w = self.resolution
        _, n_pts, _ = render_out['pts'].shape  # n_pts = number of samples per ray
        def reshape_rays_to_map(x):
            # x: (bs * h * w, c) -> (bs, c, h, w)
            c = x.shape[-1]
            assert x.shape == (bs * h * w, c), x.shape
            return x.reshape(bs, h, w, c).permute(0, 3, 1, 2)

        def render_points_to_map(x):
            # x: (bs * h * w, n_pts, c) -> (bs, c, h, w)
            c = x.shape[-1]
            assert x.shape == (bs * h * w, n_pts, c), x.shape
            return reshape_rays_to_map((x * weights_pts).sum(-2))

        def reshape_points_to_batch_points(x):
            # x: (bs * h * w, n_pts, c) -> (bs, h * w * n_pts, c)
            c = x.shape[-1]
            assert x.shape == (bs * h * w, n_pts, c), x.shape
            return x.reshape(bs, h * w * n_pts, c)

        def reshape_batch_points_to_points(x):
            # x: (bs, h * w * n_pts, c) -> (bs * h * w, n_pts, c)
            c = x.shape[-1]
            assert x.shape == (bs, h * w * n_pts, c), x.shape
            return x.reshape(bs * h * w, n_pts, c)

        weights_pts = render_out['weights'].unsqueeze(-1)  # (bs * h * w, n_pts, 1)
        weight_sum_map = reshape_rays_to_map(render_out['weight_sum'])  # (bs * h * w, 1)

        ret = dict()
        ret.update(
            weight_sum_map=weight_sum_map,
        )

        color_map = reshape_rays_to_map(render_out['color_fine'])  # already sum-weighted
        ret.update(
            color_map=color_map,
        )

        normal_pts = render_out['gradients']
        color_pts = render_out['raw_color']  # (bs * h * w, n_samples, 3)

        # del render_out['raw_color'],
        del render_out['gradients']
        light = prior_info['light']
        pts = render_out['pts']  # n_rays, n_samples, 3
        del render_out['pts']

        ambient_term = light.ambient_color[None, None, :]  # (3,) -> (1, 1, 3)
        diffuse_term = light.diffuse(normals=reshape_points_to_batch_points(normal_pts), points=reshape_points_to_batch_points(pts))  # (bs, h * w * n_pts, 3)
        if return_raw:
            ret['amb_shading_map'] = render_points_to_map(reshape_batch_points_to_points(ambient_term.expand(bs, h * w * n_pts, 3)))
            ret['diff_shading_map'] = render_points_to_map(reshape_batch_points_to_points(diffuse_term))

        shading_pts = reshape_batch_points_to_points(
            ambient_term + diffuse_term
        )
        ret['shading_map'] = render_points_to_map(shading_pts)

        if return_raw:
            ret.update(
                normal_map=render_points_to_map(normal_pts),
            )
        no_specular_pts = shading_pts * color_pts
        no_specular_map = render_points_to_map(no_specular_pts)  # (ray_bs, 1) -> (bs, 3, h, w)

        # colors = (ambient + diffuse) * texels + specular
        # https://github.com/facebookresearch/pytorch3d/blob/ea5df60d72307378d4c0641519e4e7a3671458dc/pytorch3d/renderer/mesh/shading.py#L94
        specular_term = light.specular(
            normals=reshape_points_to_batch_points(normal_pts),
            points=reshape_points_to_batch_points(pts),
            camera_position=reshape_points_to_batch_points(rays_info['rays_o'].flatten(0, 2)[:, None, :].expand_as(pts)),
        )  # (bs, h * w * n_pts, 3)
        specular_pts = reshape_batch_points_to_points(specular_term)
        specular_map = render_points_to_map(specular_pts)
        if return_raw:
            ret['no_specular_map'] = no_specular_map
            ret['specular_map'] = specular_map
        rgb_map = no_specular_map + specular_map

        bg_map = self.bg_color(bs).cuda()
        ret.update(
            image_no_bg=rgb_map,
            image=rgb_map + bg_map * (1 - weight_sum_map),  # blend rgb and bg
            mask=weight_sum_map.clamp(1e-3, 1.0 - 1e-3),
        )

        if return_raw:
            z_rays = torch.einsum('bn,bn->b', render_out['mid_z_vals'], render_out['weights']).unsqueeze(-1)  # (ray_bs, n_pts) -> (ray_bs, 1)
            ret['z_map'] = reshape_rays_to_map(z_rays)
            z_vals_min = render_out['mid_z_vals'].min(-1).values  # (ray_bs, n_pts) -> (ray_bs,)
            ret['z_min'] = z_vals_min.reshape(bs, -1).min(-1).values  # (bs,)

        return ret

    def sample_latent(self, bs, data):
        if 'w' in data:
            assert not self.training
            return {'z': data['z'], 'w': data['w']}
        if 'z' in data:
            assert not self.training
            return {'z': data['z']}
        z = torch.randn(bs, self.z_dim, device='cuda')
        return {'z': z}

    def forward(self, bs: int, it: Union[int, None], data: Dict, return_raw=False):
        if it is None:
            it = self.it.data.item()
        elif it != self.it.data.item() and it != self.it.data.item() + 1:
            import ipdb;
            ipdb.set_trace()
            if self.training and next(iter(self.parameters())).requires_grad:
                logger.error(f'inconsistent it: {it}, {self.it}')
                import ipdb; ipdb.set_trace()
            else:
                it = self.it.data.item()
        self.it.data.fill_(it)

        prior_info = self.sample_prior(bs, data)
        latent_info = self.sample_latent(bs, data)
        rays_info = self.gen_rays_at(data, prior_info)
        render_out = self.render(rays_info, latent_info=latent_info)

        new_render_out = self.render_maps(
            bs=bs, render_out=render_out,
            rays_info=rays_info, prior_info=prior_info, return_raw=return_raw
        )
        loss = {
            'eikonal': render_out['gradient_error'],
        }
        stats = {
            'surface': render_out['surface_loss'],
            's_val': render_out['s_val'].mean(),
            'cdf': render_out['cdf_fine'][:, :1].mean(),
            'weight_max': render_out['weight_max'].mean(),
            'weight_sum': render_out['weight_sum'].mean(),
        }

        blob = {'loss': loss, 'stats': stats}
        blob['stats'][f'light/ambient'] = self.light.ambient_color.mean().item()
        blob['stats'][f'light/diffuse'] = self.light.diffuse_color.mean().item()
        blob['stats'][f'light/specular'] = self.light.specular_color.mean().item()
        blob['stats'][f'material/shininess'] = self.light.shininess.item()
        blob.update({'render_out': new_render_out, 'prior_info': prior_info})
        if return_raw:
            blob.update({
                'latent_info': latent_info,
                'rays_info': rays_info,
                'raw_render_out': render_out,
            })
        return {'box': blob}

    def render_one_chunk(self, rays_o, rays_d, latent_info, **kwargs):
        near, far = near_far_from_sphere(rays_o, rays_d)
        if 'w' not in latent_info:
            z = latent_info['z']
            w = self.renderer.sdf_network.style(z)
            latent_info['w'] = w
        else:
            if os.getenv('RECON') != '1' or os.getenv('RECON_OBJ') == '1':
                assert not self.training
            z = latent_info['z']
            w = latent_info['w']
        cos_anneal_ratio = np.min([1.0, self.it.item() / self.anneal_end])
        render_out = self.renderer.render(
            rays_o, rays_d, near, far,
            background_rgb=None,
            cos_anneal_ratio=cos_anneal_ratio,
            perturb_overwrite=-1 if self.training else 0,
            z=z, w=w,
            **kwargs
        )
        return render_out

    def gen_rays_at(self, data, prior_info: Dict) -> Dict[str, torch.Tensor]:
        b2w = prior_info['b2w']
        b2c = torch.einsum('ij,bjk->bik', self.camera.w2c, b2w)
        b2c_trans = b2c[..., :3, 3]

        """ crop around the box """

        center_x = self.camera.cam_dist / b2c_trans[..., 2] * b2c_trans[..., 0] * self.resolution / 2 + 1 / 2 * self.scene_resolution
        center_y = self.camera.cam_dist / b2c_trans[..., 2] * b2c_trans[..., 1] * self.resolution / 2 + 1 / 2 * self.scene_resolution

        x_offset = center_x - self.resolution / 2
        y_offset = center_y - self.resolution / 2

        rays_v = build_rays(
            h_recp_size=self.resolution, w_recp_size=self.resolution,
            h_offset=y_offset, w_offset=x_offset,
            num_rays_h=self.resolution, num_rays_w=self.resolution,
            intrinsics=self.camera.intrinsics, intrinsics_inv=self.camera.intrinsics_inv,
        )

        # from camera to world to box frame
        c2b = prior_info['c2b']
        rays_v = torch.einsum('bij,bhwj->bhwi', c2b[..., :3, :3], rays_v)  # (b, h, w, 3)
        rays_o = c2b[:, None, None, :3, 3].expand(rays_v.shape)  # (b, h, w, 3)
        return {'rays_o': rays_o, 'rays_d': rays_v, 'x_offset': x_offset, 'y_offset': y_offset}

    def render(self, rays_info, **kwargs):
        rays_o = rays_info['rays_o']
        rays_d = rays_info['rays_d']
        bs = rays_o.shape[0]

        max_ray_bs = MAX_RAY_BATCH_SIZE
        chunk_size = int(max_ray_bs / bs)
        num_chunks = math.ceil(rays_o.flatten(1, 2).shape[1] / chunk_size)
        if num_chunks > 1:
            assert not self.training, (rays_o.shape, chunk_size, num_chunks, max_ray_bs)
        render_out = None
        for chunk_ind in range(num_chunks):
            rays_o_chunk = rays_o.flatten(1, 2)[:, chunk_ind * chunk_size:(chunk_ind + 1) * chunk_size].flatten(0, 1)
            rays_d_chunk = rays_d.flatten(1, 2)[:, chunk_ind * chunk_size:(chunk_ind + 1) * chunk_size].flatten(0, 1)
            chunk_out = self.render_one_chunk(rays_o_chunk, rays_d_chunk, **kwargs)
            if render_out is None:
                render_out = {k: [] for k in chunk_out.keys() if
                              k not in ['gradient_error', 'divergence_reg', 'surface_loss']}
            for k in chunk_out.keys():
                if k in ['gradient_error', 'divergence_reg', 'surface_loss']:
                    # handle separately
                    continue
                render_out[k].append(chunk_out[k].unflatten(0, (bs, -1)))
        for k in render_out.keys():
            render_out[k] = torch.cat(render_out[k], dim=1).flatten(0, 1)  # (bs, h * w, ...) -> (ray_bs = bs * h * w, ...)

        if num_chunks == 1:
            render_out['gradient_error'] = chunk_out['gradient_error']
            render_out['surface_loss'] = chunk_out['surface_loss']
        else:
            assert not self.training
            render_out['gradient_error'] = None  # hack
            render_out['surface_loss'] = None  # hack
        return render_out


def build_rays(
        *,
        h_recp_size: int, w_recp_size: int,  # they can technically be tensors of shape (bs,)
        h_offset: torch.Tensor, w_offset: torch.Tensor,
        num_rays_h: int, num_rays_w: int,
        intrinsics: torch.Tensor, intrinsics_inv: torch.Tensor,
):
    tx = torch.linspace(0, 1, num_rays_w, device=intrinsics.device)
    ty = torch.linspace(0, 1, num_rays_h, device=intrinsics.device)
    pixels_x, pixels_y = torch.meshgrid(tx, ty, indexing='ij')  # (w, h)
    pixels_x = pixels_x * w_recp_size + w_offset[..., None, None]  # (..., w, h)
    pixels_y = pixels_y * h_recp_size + h_offset[..., None, None]  # (..., w, h)
    p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # ..., w, h, 3
    p = torch.einsum('ij,...whj->...whi', intrinsics_inv[:3, :3], p)  # ..., w, h, 3
    p = torch.einsum('...whi->...hwi', p)
    rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)
    return rays_v


def near_far_from_sphere(rays_o, rays_d):
    a = torch.sum(rays_d**2, dim=-1, keepdim=True)
    b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
    mid = 0.5 * (-b) / a
    near = mid - 1.0
    far = mid + 1.0
    return near, far
