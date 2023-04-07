import torch.nn as nn
import numpy as np
from ..utils.pose import get_identity_pose
import torch
import logging
logger = logging.getLogger(__name__)


class Camera(nn.Module):
    def __init__(self, cam_dist, fov, resolution):
        super().__init__()
        self.resolution = resolution
        self.cam_dist = cam_dist

        focal = (resolution / 2) * 1 / np.tan(0.5 * fov * np.pi / 180.)
        intrinsics = torch.tensor([
            [focal, 0, 0.5 * resolution, 0],
            [0, focal, 0.5 * resolution, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=torch.float32)
        self.register_buffer('intrinsics', intrinsics)
        self.register_buffer('intrinsics_inv', torch.tensor(np.linalg.inv(intrinsics.numpy()), dtype=torch.float32))
        c2w, w2c = get_identity_pose(cam_dist=cam_dist)
        self.register_buffer('c2w', c2w)
        self.register_buffer('w2c', w2c)

        logger.info(f'create camera with focal: {focal} and cam_dist: {cam_dist}')
