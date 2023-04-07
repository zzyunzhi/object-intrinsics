# for lighting visualization
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch


class UnitSphere(nn.Module):
    def __init__(self, resolution, scale=1):
        super().__init__()
        assert 0 <= scale <= 1, scale
        self.resolution = resolution
        xx, yy = np.meshgrid(
            np.linspace(-(1 - .5 / resolution), 1 - .5 / resolution, resolution),
            np.linspace(-(1 - .5 / resolution), 1 - .5 / resolution, resolution)
        )
        # orthographic camera
        self.register_buffer(
            'points_grid',
            torch.tensor(np.stack([xx, yy], axis=-1), dtype=torch.float32).reshape(-1, 2)
        )
        # shape (res * res, 2), in range [-1, 1]

        xx = xx.clip(-1 * scale, 1 * scale)
        yy = yy.clip(-1 * scale, 1 * scale)
        zz = -np.sqrt(scale ** 2 - xx ** 2 - yy ** 2)  # a sphere with radius = scale
        self.register_buffer(
            'points_box',
            torch.tensor(np.stack([xx, yy, zz], axis=-1), dtype=torch.float32).reshape(-1, 3)  # (res * res, 3)
        )
        self.register_buffer(
            'points_mask',
            ~torch.isnan(self.points_box[:, 2])  # (res * res,)
        )

    def render(self, light):
        # light is in the box frame
        ret = {}
        if hasattr(light, 'batch_size'):
            bs = light.batch_size
            points = self.points_box[None].expand(bs, -1, -1)
        else:
            points = self.points_box
            bs = 1
        shading = light.ambient_color + light.diffuse(normals=points, points=points)  # (bs, res * res, 3)

        shading = shading.reshape(bs, -1, 3)
        shading[~self.points_mask[None, :, None].expand(bs, -1, 3)] = 0

        shading_map = F.grid_sample(
            shading.reshape(bs, self.resolution, self.resolution, 3).permute(0, 3, 1, 2),
            grid=self.points_grid.reshape(1, self.resolution, self.resolution, 2).expand(bs, -1, -1, -1),
            padding_mode='zeros',
            align_corners=False
        )
        ret['shading_map'] = shading_map
        if not hasattr(light, 'batch_size'):
            ret['shading_map'] = ret['shading_map'].squeeze(0)
        return ret
