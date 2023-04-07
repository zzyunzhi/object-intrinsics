import os
import cv2
import numpy as np


def cv2_read_rgb(path, size=None):
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError(f'failed to read {path}')
    assert arr.shape[2] == 3, arr.shape
    assert arr.dtype == np.uint8, arr.dtype
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)  # uint8 (h, w, 3)
    if size is not None:
        arr = cv2.resize(arr, size, interpolation=cv2.INTER_LINEAR)
    return arr


def cv2_read_grayscale(path, size=None):
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    assert len(arr.shape) == 2, arr.shape
    assert arr.dtype == np.uint8, arr.dtype
    if size is not None:
        arr = cv2.resize(arr, size, interpolation=cv2.INTER_LINEAR)
    return arr


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


def cv2_write_rgba(arr: np.ndarray, path: str):
    assert len(arr.shape) == 3 and arr.dtype == np.uint8 and arr.shape[2] == 4, (arr.dtype, arr.shape)
    assert os.path.exists(os.path.dirname(path)), path
    cv2.imwrite(path, cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA))


def cv2_write_mask(arr: np.ndarray, path: str):
    assert len(arr.shape) == 2 and arr.dtype == np.bool, (arr.dtype, arr.shape)
    assert os.path.exists(os.path.dirname(path)), path
    cv2.imwrite(path, (arr * 255).astype(np.uint8))


def cv2_write_rgb(arr: np.ndarray, path: str):
    assert len(arr.shape) == 3 and arr.dtype == np.uint8 and arr.shape[2] == 3, (arr.dtype, arr.shape)
    assert os.path.exists(os.path.dirname(path)), path
    cv2.imwrite(path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


def cv2_resize(arr: np.ndarray, max_size: int) -> np.ndarray:
    # resize but keep aspect ratio
    assert len(arr.shape) in [2, 3], arr.shape
    ratio = max_size / max(arr.shape[:2])
    return cv2.resize(arr, (int(arr.shape[1] * ratio), int(arr.shape[0] * ratio)), interpolation=cv2.INTER_LINEAR)
