import numpy as np
import torch
from typing import Callable, Union, Tuple, List
from ..models.lighting import DirectionalLightWithSpecularFixInit
from .pose import look_at
import logging

logger = logging.getLogger(__name__)


def load_bg_color_fn(bg_color_str, h, w) -> Callable:
    def fn(bs: int, bg_color: Union[None, torch.Tensor] = None) -> torch.Tensor:
        if bg_color is None:
            if bg_color_str == 'random':
                arr = np.random.uniform(low=0, high=1, size=(bs, 3))
            elif bg_color_str == 'black':
                arr = np.array([0, 0, 0])[None]
            elif bg_color_str == 'white':
                arr = np.array([1, 1, 1])[None]
            else:
                raise NotImplementedError(bg_color_str)
            arr = torch.tensor(arr, dtype=torch.float32)
        else:
            assert bg_color.shape == (bs, 3)
            arr = bg_color
        arr = arr[:, :, None, None].expand(bs, 3, h, w)
        return arr
    return fn


def build_directional_light_optimizable(cam_loc: Union[None, np.ndarray, Tuple, List], light_loc: Union[None, np.ndarray, Tuple, List],
                                        ambient_color=0.33, diffuse_color=0.66, specular_color=0, shininess=10):
    if cam_loc is None and light_loc is None:
        # collocated light
        cam_loc = [0, 0, -1]
        light_loc = [0, 0, -1]
    dw = np.array(light_loc) / np.linalg.norm(light_loc)
    c2w = look_at(cam_loc).numpy()
    dc = c2w.T @ dw
    direction = dc
    logger.info(f'build light with direction: {direction}')

    return DirectionalLightWithSpecularFixInit(
        direction=direction,
        ambient_color=ambient_color,
        diffuse_color=diffuse_color,
        specular_color=specular_color,
        shininess=shininess,
    )
