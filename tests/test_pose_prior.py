import numpy as np
import matplotlib.pyplot as plt
import torch
from src.utils.pose_sampler import Hemisphere, Sphere
from src.utils.plot import plot_camera_scene
import logging
import matplotlib
matplotlib.use('TkAgg')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_prior():
    # There are multiple instances in the scene, captured with a fixed camera viewpoint.
    prior = Sphere()
    mat = prior(100)
    # mat[0, :3, :3] = prior.c2w_canonical.T  # FIXME: hack
    mat[0, :3, :3] = np.eye(3)
    mat = torch.tensor(mat, dtype=torch.float32)

    trans = torch.einsum('bij,j->bi', mat[:, :3, :3], torch.tensor([0, 0, 1], dtype=torch.float32))
    mat[:, :3, 3] = trans

    fig, ax = plot_camera_scene(c2w=mat[1:], c2w_gt=mat[:1], plot_radius=1., return_fig=True)
    plt.show()


def test_multiview_prior():
    # There is only one instance in the scene, captured from multiple viewpoints.

    prior = Hemisphere()
    mat = prior(100)
    mat[0, :3, :3] = np.eye(3)
    mat = torch.tensor(mat, dtype=torch.float32)

    mat = torch.einsum('bij->bji', mat)  # b2w -> w2b = c2w
    trans = torch.einsum('bij,j->bi', mat[:, :3, :3], torch.tensor([0, 0, -1], dtype=torch.float32))
    mat[:, :3, 3] = trans

    fig, ax = plot_camera_scene(c2w=mat[1:], c2w_gt=mat[:1], plot_radius=1., return_fig=True)
    plt.show()


if __name__ == "__main__":
    test_prior()
    # test_multiview_prior()
