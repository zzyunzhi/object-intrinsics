import os.path
import torch.distributed as dist
import torch.nn as nn
import torch
from ..third_party.stylesdf.volume_renderer import SirenGenerator
from ..third_party.stylesdf.model import MappingLinear
from tu.utils.config import check_cfg_consistency


class ShapeNetwork(nn.Module):
    def __init__(self, checkpoint_path, **kwargs):
        super().__init__()

        layers = []
        for i in range(3):
            layers.append(
                MappingLinear(kwargs['style_dim'], kwargs['style_dim'], activation="fused_lrelu")
            )
        style = nn.Sequential(*layers)
        network = SirenGenerator(**kwargs)
        self.style = style
        self.pts_linears = network.pts_linears
        self.sigma_linear = network.sigma_linear

        if checkpoint_path is not None:
            # logger.info(f'loading from {checkpoint_path}')

            # hack
            if dist.is_initialized():
                device = f'cuda:{dist.get_rank()}'
                self.to(device)
            else:
                device = 'cuda'
            # logger.info(f'loading to device: {device}')
            state_dict = torch.load(checkpoint_path, map_location=device)
            check_cfg_consistency(kwargs, state_dict['cfg']['model']['generator']['kwargs']['sdf_network']['kwargs'],
                                  ignore_keys=['checkpoint_path',])
            self.load_state_dict(state_dict['sdf_network'])

        # g = nn.Module()
        # g.style = style
        # g.renderer = nn.Module()
        # g.renderer.network = network
        #
        # path = './intrinsics/third_party/stylesdf/pretrained_renderer/sphere_init.pt'
        # state_dict = torch.load(path)
        # g.load_state_dict(state_dict['g'], strict=False)

    def forward(self, x, z, w=None):
        ray_bs = x.shape[0]
        if w is not None:
            bs = w.shape[0]
        else:
            bs = z.shape[0]
        x = x.reshape(bs, ray_bs//bs, 1, 1, *x.shape[1:])

        if w is None:
            latent = self.style(z)
        else:
            latent = w

        mlp_out = x.contiguous()
        for i in range(len(self.pts_linears)):
            mlp_out = self.pts_linears[i](mlp_out, latent)

        sdf = self.sigma_linear(mlp_out)
        # print(torch.linalg.norm(sdf[:, :2], dim=-1).flatten(), torch.linalg.norm(x[:, :2], dim=-1).flatten())

        outputs = torch.cat([sdf, mlp_out], -1)
        return outputs.flatten(0, 3)

    def sdf(self, x, z, w=None):
        return self.forward(x, z=z, w=w)[:, :1]

    def gradient(self, x, z, w=None, second_order=False):
        return gradient(x, lambda pts: self.sdf(pts, z=z, w=w).squeeze(-1),
                        second_order=second_order)


class ColorNetwork(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        network = SirenGenerator(**kwargs)
        self.views_linears = network.views_linears
        self.rgb_linear = network.rgb_linear
        self.style_dim = kwargs['style_dim']
        self.w_dim = kwargs['W']

    def forward(self, points, normals, view_dirs, feature_vectors, z, w):
        ray_bs = points.shape[0]
        bs = w.shape[0]
        feature_vectors = feature_vectors.reshape(bs, ray_bs // bs, 1, 1, *feature_vectors.shape[1:])
        normals = normals.reshape(bs, ray_bs // bs, 1, 1, *normals.shape[1:])

        mlp_out = torch.cat([feature_vectors, normals], -1)
        out_features = self.views_linears(mlp_out, w)
        rgb = self.rgb_linear(out_features)
        rgb = torch.sigmoid(rgb)

        rgb = rgb.flatten(0, 3)
        return rgb


def gradient(x, fn, second_order=False, laplacian=False):
    has_grad_outer = torch.is_grad_enabled()
    x.requires_grad_(True)
    if not has_grad_outer:
        torch.set_grad_enabled(True)
    y = fn(x)  # scalar function
    if isinstance(y, tuple):
        x, y = y
    assert y.shape == x.shape[:-1], (y.shape, x.shape)
    if not has_grad_outer and not second_order and not laplacian:
        torch.set_grad_enabled(False)
    d_output = torch.ones_like(y, requires_grad=False, device=y.device)
    gradients = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=d_output,
        create_graph=has_grad_outer or second_order or laplacian,
        retain_graph=has_grad_outer or second_order or laplacian,
    )[0]
    if second_order or laplacian:
        hess = []
        for ind in range(gradients.shape[-1]):
            gradients_slice = gradients[..., ind]

            if ind == gradients.shape[-1] - 1:
                if not has_grad_outer:
                    torch.set_grad_enabled(False)

            d_output = torch.ones_like(gradients_slice, requires_grad=False, device=y.device)
            hess_slice = torch.autograd.grad(
                outputs=gradients_slice,
                inputs=x,
                grad_outputs=d_output,
                create_graph=has_grad_outer or ind < gradients.shape[-1] - 1,
                retain_graph=has_grad_outer or ind < gradients.shape[-1] - 1,
            )[0]
            hess.append(hess_slice)
        hess = torch.stack(hess, dim=-1)
        if second_order:
            return gradients, hess
        # else return laplacian
        return gradients, torch.diagonal(hess, dim1=-2, dim2=-1)
    return gradients
