import cv2
import numpy as np


def cv2_read_rgba(path, mask_threshold=128, assert_binary=True, size=None):
    # size is (w, h)
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError(f'failed to read {path}')

    assert arr.shape[2] == 4, arr.shape
    assert arr.dtype == np.uint8, arr.dtype
    arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)  # uint8 (h, w, 4)
    if assert_binary:
        assert np.logical_or(arr[:, :, 3] == 0, arr[:, :, 3] == 255).all(), (np.unique(arr[:, :, 3]), path)
    if size is not None:
        arr = cv2.resize(arr, size, interpolation=cv2.INTER_LINEAR)
    rgb = arr[:, :, :3]  # uint8 (h, w, 3)
    mask = arr[:, :, 3] >= mask_threshold  # boolean (h, w)
    return arr, rgb, mask
