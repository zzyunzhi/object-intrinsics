import os
import numpy as np
from typing import Tuple, Union
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import logging

logger = logging.getLogger(__name__)


def look_at(eye, center=(0, 0, 0), up=(0, 1, 0), skip_check=False):
    if isinstance(eye, (tuple, list)):
        eye = torch.tensor(eye, dtype=torch.float32)
    device = eye.device
    if isinstance(center, (tuple, list)):
        center = torch.tensor(center, dtype=torch.float32, device=device)
    if isinstance(up, (tuple, list)):
        up = torch.tensor(up, dtype=torch.float32, device=device)

    vec_forward = F.normalize(center - eye, p=2, dim=-1)
    up = F.normalize(up, p=2, dim=-1)

    # https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function
    if torch.allclose(vec_forward, torch.tensor([0, 1, 0], dtype=torch.float32, device=device)) and \
            torch.allclose(up, torch.tensor([0, 1, 0], dtype=torch.float32, device=device)):
        logger.warning(f'special case for look at: eye {eye}, center {center}, up {up}')
        # import ipdb; ipdb.set_trace()
        return torch.tensor([
            # changed after 10/01/2022
            # [1., 0., 0.],  # assume that x-direction of camera is consistent with world, i.e. x is right
            # [0., 0., 1.],
            # [0., 1., 0.],

            # changed after 10/04/2022
            [1., 0., 0.],  # assume that x-direction of camera is consistent with world, i.e. x is right
            [0., 0., 1.],
            [0., -1., 0.],
        ], dtype=torch.float32, device=device)

    up = up.expand_as(vec_forward)
    vec_right = up.cross(vec_forward, dim=-1)
    vec_right = F.normalize(vec_right, p=2, dim=-1)
    if torch.linalg.norm(vec_right) == 0:
        if not skip_check:
            import ipdb; ipdb.set_trace()
        else:
            logger.error(f'invalid vec_right: {vec_right}')

    vec_up = vec_forward.cross(vec_right, dim=-1)
    vec_up = F.normalize(vec_up, p=2, dim=-1)
    if torch.linalg.norm(vec_up) == 0:
        if not skip_check:
            import ipdb; ipdb.set_trace()
        else:
            logger.error(f'invalid vec_up: {vec_up}')

    rot_mat = torch.stack([vec_right, vec_up, vec_forward], -1)
    check_rot_mat(rot_mat)
    return rot_mat


def get_lookat_mat(eye, center=(0, 0, 0)):
    if isinstance(eye, tuple):
        eye = torch.tensor(eye, dtype=torch.float32)
    rot = look_at(eye, center=center)
    mat = assemble_rot_trans(rot, eye)
    return mat


def opengl_to_colmap(pose):
    # https://github.com/Totoro97/NeuS/blob/2708e43ed71bcd18dc26b2a1a9a92ac15884111c/preprocess_custom_data/colmap_preprocess/pose_utils.py#L51
    # colmap https://colmap.github.io/format.html
    # opengl: [r, u, -t]
    # colmap / neus: [r, -u, t]
    # llff: [-u, r, -t]
    convert = torch.tensor([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ], dtype=torch.float32, device=pose.device)
    # return pose @ convert[:3, :3]
    return convert[:3, :3] @ pose


def to_sphere(u, v):
    theta = 2 * np.pi * u
    phi = np.arccos(1 - 2 * v)
    cx = np.sin(phi) * np.cos(theta)
    cy = np.sin(phi) * np.sin(theta)
    cz = np.cos(phi)
    return np.stack([cx, cy, cz], axis=-1)


def sample_on_sphere(range_u=(0, 1), range_v=(0, 1), size=(1,),
                     to_pytorch=True):
    u = np.random.uniform(*range_u, size=size)
    v = np.random.uniform(*range_v, size=size)

    sample = to_sphere(u, v)
    if to_pytorch:
        sample = torch.tensor(sample, dtype=torch.float32)

    return sample


def get_camera_mat(res, fov):
    # fov = 2 * arctan( sensor / (2 * focal))
    # focal = (sensor / 2)  * 1 / (tan(0.5 * fov))

    # pixels are in range [0, res - 1]

    focal = ((res - 1) / 2) * 1 / np.tan(0.5 * fov * np.pi / 180.)

    intrinsic = torch.tensor([
        [focal, 0, 0.5 * (res - 1), 0],
        [0, focal, 0.5 * (res - 1), 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32)
    return intrinsic


def get_camera_mat_v2(res, fov):
    # fov = 2 * arctan( sensor / (2 * focal))
    # focal = (sensor / 2)  * 1 / (tan(0.5 * fov))

    # pixels are in range [0, res - 1]

    focal = (res / 2) * 1 / np.tan(0.5 * fov * np.pi / 180.)

    intrinsic = torch.tensor([
        [focal, 0, 0.5 * res, 0],
        [0, focal, 0.5 * res, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32)
    return intrinsic


def invert_rot_t(pose):
    # https://ksimek.github.io/2012/08/22/extrinsic/
    assert pose.shape[-2:] in [(3, 4), (4, 4)], pose.shape
    # assume that pose[..., :3, :3] is rotation
    # pose[..., :3, 3] is translation
    # the rest is [0, 0, 0, 1] for padding
    rot = pose[..., :3, :3]
    t = pose[..., :3, 3]

    pose_inv_rot = torch.transpose(rot, -2, -1)
    pose_inv_trans = -torch.matmul(pose_inv_rot, t.unsqueeze(-1)).squeeze(-1)
    return assemble_rot_trans(pose_inv_rot, pose_inv_trans)


def invert_rot_t_np(pose):
    assert pose.shape[-2:] in [(3, 4), (4, 4)], pose.shape
    R = pose[..., :3, :3]
    t = pose[..., :3, 3]
    pose_inv_R = np.transpose(R, (-1, -2))
    pose_inv_t = -np.einsum('...ij,...j->...i', pose_inv_R, t)
    return assemble_rot_trans_np(pose_inv_R, pose_inv_t)


def invert_rot_trans(rot, trans):
    # rot: (..., 3, 3)
    # trans: (..., 3)
    assert rot.shape[-2:] == (3, 3), (rot.shape, trans.shape)
    assert trans.shape[-1] == 3, (rot.shape, trans.shape)
    assert rot.shape[:-2] == trans.shape[:-1], (rot.shape, trans.shape)

    new_rot = torch.transpose(rot, -2, -1)
    new_trans = -torch.matmul(new_rot, trans.unsqueeze(-1)).squeeze(-1)
    return new_rot, new_trans


def assemble_rot_trans_np(rot, trans):
    assert rot.shape[-2:] == (3, 3)
    pose = np.concatenate([rot, trans.reshape(*rot.shape[:-2], 3, 1)], axis=-1)  # (..., 3, 4)
    return mat_34_to_44_np(pose)


def assemble_rot_trans(rot, trans):
    assert rot.shape[-2:] == (3, 3)
    pose = torch.cat([rot, trans.reshape(*rot.shape[:-2], 3, 1)], dim=-1)  # (..., 3, 4)
    return mat_34_to_44_torch(pose)


def get_identity_pose(cam_dist=1.):
    # cam_dist is in coordinate system unit
    assert cam_dist > 0
    c2w_t = torch.tensor([0, 0, -1], dtype=torch.float32)
    c2w_rot = look_at(c2w_t)
    c2w_t = cam_dist * c2w_t
    c2w = torch.eye(4, dtype=torch.float32)
    c2w[:3, :3] = c2w_rot
    c2w[:3, 3] = c2w_t

    if False:  ## FIXME: DEBUG
        w2c = torch.eye(4, dtype=torch.float32)
        w2c[:3, :3] = c2w_rot.transpose(0, 1)
        w2c[:3, 3] = -torch.matmul(c2w_rot.transpose(0, 1), c2w_t.unsqueeze(-1)).squeeze(-1)
        assert torch.allclose(w2c, invert_rot_t(c2w))

    return c2w, invert_rot_t(c2w)


def get_blender_identity_pose(cam_dist=1.):
    assert cam_dist > 0
    c2w_t = torch.tensor([0, 0, -1], dtype=torch.float32)
    c2w_rot = torch.tensor([
        [1., 0., 0.],
        [0., -1., 0.],
        [0., 0., -1.],
    ], dtype=torch.float32)
    c2w_t = cam_dist * c2w_t
    c2w = assemble_rot_trans(c2w_rot, c2w_t)
    return c2w, invert_rot_t(c2w)


def get_identity_obj_rot(return_eye=False):
    b2w_t = torch.tensor([0, 0, 1], dtype=torch.float32)
    b2w_rot = look_at(b2w_t)
    if return_eye:
        return b2w_rot, b2w_t
    return b2w_rot


def get_random_pose(batch_size, range_u=(0, 1), range_v=(0, 1), return_eye=False):
    eye = sample_on_sphere(range_u=range_u, range_v=range_v, size=(batch_size,))
    b2w = look_at(eye)
    if return_eye:
        return b2w, eye
    return b2w


def rejection_sample_transform(batch_size, accept_fn, up, max_tries=100, to_44=False):
    r_all = np.zeros((0, 3, 3))
    if isinstance(up, tuple):
        up = up.array(up)
    assert up.shape == (3,), up.shape

    while r_all.shape[0] < batch_size:
        r = R.random(batch_size * 10).as_matrix()
        pts = (r @ up.reshape(3, 1)).squeeze(-1)
        r = r[accept_fn(pts), :, :]
        r_all = np.concatenate([r_all, r], axis=0)
        max_tries -= 1
        if max_tries == 0:
            print('infinite loop')
            import ipdb;
            ipdb.set_trace()

    r_all = r_all[:batch_size, :, :]
    if to_44:
        return mat_33_to_44_np(r_all)
    return r_all


def mat_33_to_44_np(x):
    bs = np.prod(x.shape[:-2])
    assert x.shape[-2:] == (3, 3)
    template = np.tile(np.eye(4).reshape(1, 4, 4), (bs, 1, 1))
    template = template.reshape(*x.shape[:-2], 4, 4)
    template[..., :3, :3] = x
    return template


def check_rot_mat(x):
    prod = x @ torch.transpose(x, -2, -1)
    if len(prod.shape) == 2:
        prod = prod[None, :, :]
    if len(prod.shape) > 3:
        prod = prod.flatten(start_dim=0, end_dim=-3)
    if not torch.allclose(
            prod, torch.eye(prod.shape[-1], device=x.device)[None, :, :].expand(prod.shape[0], -1, -1), atol=1e-2):
        import ipdb;
        ipdb.set_trace()
        return False
    return True

def check_rot_mat_np(x):
    # assume that x is a rotation matrix
    # with shape (..., 3, 3) or (..., 4, 4)
    prod = x @ np.transpose(x, (-1, -2))
    if len(prod.shape) == 2:
        prod = prod[None, :, :]
    if not np.allclose(prod, np.eye(prod.shape[-1])[None, :, :].repeat(prod.shape[0], axis=0), atol=1e-2):
        import ipdb;
        ipdb.set_trace()
        return False
    return True

def check_rot_scaling_mat_np(x):
    scale = np.linalg.norm(x, axis=-2, keepdims=True)
    rot = np.divide(x, scale)
    return check_rot_mat_np(rot)


def mat_34_to_44_np(x):
    bs = np.prod(x.shape[:-2], dtype=np.int32)
    pad_down = np.array([0, 0, 0, 1.]).reshape(1, 1, 4)
    pad_down = np.tile(pad_down, (bs, 1, 1))
    x = np.concatenate([x.reshape(-1, 3, 4), pad_down], axis=-2).reshape((*x.shape[:-2], 4, 4))
    return x


def mat_34_to_44_torch(x):
    bs = np.prod(x.shape[:-2], dtype=np.int32)
    pad_down = torch.tensor([0., 0., 0., 1.], device=x.device, dtype=x.dtype).reshape(1, 1, 4).repeat(bs, 1, 1)
    x = torch.cat([x.reshape(bs, 3, 4), pad_down], dim=-2).reshape(*x.shape[:-2], 4, 4)
    return x


def mat_33_to_44_torch(x):
    bs = np.prod(x.shape[:-2], dtype=np.int32)
    assert x.shape[-2:] == (3, 3)

    pad_right = torch.tensor([0., 0., 0.], device=x.device, dtype=x.dtype).reshape(1, 3, 1).repeat(bs, 1, 1)

    x = torch.cat([x.reshape(bs, 3, 3), pad_right], dim=-1).reshape(*x.shape[:-2], 3, 4)
    return mat_34_to_44_torch(x)

    template = torch.eye(4, device=x.device).reshape(1, 4, 4).repeat(bs, 1, 1)
    template = template.reshape(*x.shape[:-2], 4, 4)
    # FIXME: does gradient propagate?
    template[..., :3, :3] = x
    return template


# estimate pose based
def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def get_b2w_rot(endpoint_w: Union[Tuple, np.ndarray],
                endpoint_b: Union[Tuple, np.ndarray]):
    if isinstance(endpoint_w, tuple):
        endpoint_w = np.array(endpoint_w)
    if isinstance(endpoint_b, tuple):
        endpoint_b = np.array(endpoint_b)
    assert endpoint_w.shape == (3,), endpoint_w.shape
    assert endpoint_b.shape == (3,), endpoint_b.shape
    if np.allclose(endpoint_w, endpoint_b):
        return np.eye(4)
    rot_ax = np.cross(endpoint_b, endpoint_w)
    rot_ax = rot_ax / np.linalg.norm(rot_ax)

    rot_ang = angle_between(endpoint_w, endpoint_b)
    b2w_rot = R.from_rotvec(rot_ax * rot_ang).as_matrix()

    assert np.allclose((b2w_rot @ endpoint_b.reshape(3, 1)).squeeze(-1),
                       endpoint_w / np.linalg.norm(endpoint_w))

    b2w_rot = mat_33_to_44_np(b2w_rot)
    return b2w_rot


def interpolate_pose(c2w_0, c2w_1, ratio):
    pose_0 = c2w_0.detach().cpu().numpy()
    pose_1 = c2w_1.detach().cpu().numpy()
    pose_0 = np.linalg.inv(pose_0)
    pose_1 = np.linalg.inv(pose_1)
    rot_0 = pose_0[:3, :3]
    rot_1 = pose_1[:3, :3]
    rots = R.from_matrix(np.stack([rot_0, rot_1]))
    key_times = [0, 1]
    slerp = Slerp(key_times, rots)
    rot = slerp(ratio)
    pose = np.diag([1.0, 1.0, 1.0, 1.0])
    pose = pose.astype(np.float32)
    pose[:3, :3] = rot.as_matrix()
    pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
    pose = np.linalg.inv(pose)
    return torch.tensor(pose, dtype=torch.float32, device='cuda')


def rotation_matrix_from_vectors(vec1, vec2):
    # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    if np.allclose(a, b):
        return np.eye(3)
    if np.allclose(a, -b):
        # the rotation matrix is not unique
        raise NotImplementedError
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def pose_to_d9(pose: torch.Tensor) -> torch.Tensor:
    assert pose.shape[-2:] == (4, 4), pose.shape
    R = pose[..., :3, :3]  # (..., 3, 3)
    t = pose[..., :3, 3]  # (..., 3)
    r6 = R[..., :2, :3].flatten(-2, -1)  # (..., 6)
    d9 = torch.cat([t, r6], -1)  # [N, 9]
    return d9


def r6d2mat(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalisation per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    assert d6.shape[-1] == 6, d6.shape
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def d9_to_pose(d9: torch.Tensor) -> torch.Tensor:
    assert d9.shape[-1] == 9, d9.shape
    t = d9[..., :3]  # (..., 3)
    r6 = d9[..., 3:]  # (..., 6)
    R = r6d2mat(r6)  # (..., 3, 3)
    return assemble_rot_trans(R, t)


def get_tip_from_spherical_coord(elev: np.ndarray, azim: np.ndarray) -> np.ndarray:
    # elev, azim: (...,)
    z = -np.sin(elev) * np.cos(azim)
    x = np.sin(elev) * np.sin(azim)
    y = -np.cos(elev)
    return np.stack([x, y, z], axis=-1)
