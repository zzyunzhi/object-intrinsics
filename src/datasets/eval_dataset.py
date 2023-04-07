import os
import glob
import numpy as np
from ..utils.prior import load_bg_color_fn
from ..utils.preprocess import cv2_read_rgba
import torch
import logging


logger = logging.getLogger(__name__)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, resolution, dataset_folder):
        super().__init__()
        self.resolution = resolution
        self.dataset_folder = dataset_folder
        paths = list(sorted(glob.glob(os.path.join(dataset_folder, '*.png'))))
        logger.info(f'found {len(paths)} images in {dataset_folder}')
        self.num_images = len(paths)

        rgb_list = []
        mask_list = []
        for path in paths:
            rgba, rgb, mask = cv2_read_rgba(path, size=(self.resolution, self.resolution), assert_binary=False)
            rgb_list.append(rgb)
            mask_list.append(mask)

        self.data = {
            'rgb': torch.tensor(np.stack(rgb_list, axis=0), dtype=torch.float32).permute(0, 3, 1, 2) / 255.,  # (n_images, 3, h, w)
            'alpha': torch.tensor(np.stack(mask_list, axis=0), dtype=torch.float32)[:, None, :, :],  # (n_images, 1, h, w)
            'path': paths,
        }

        bg_color_fn = load_bg_color_fn('random', self.resolution, self.resolution)
        self.bg_color_fn = lambda: bg_color_fn(bs=1).squeeze(0)

    def __getitem__(self, index):
        rgb = self.data['rgb'][index]
        alpha = self.data['alpha'][index]

        rgb = rgb * alpha + self.bg_color_fn() * (1 - alpha)
        data = {
            'image': rgb,
            'mask': alpha,
            'image_path': self.data['path'][index],
            'pose_indices': index,
        }
        return data

    def __len__(self):
        return self.num_images
