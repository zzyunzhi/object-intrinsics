import numpy as np
from typing import Union, List
from .pose import assemble_rot_trans_np, mat_33_to_44_np, get_lookat_mat, look_at
from scipy.spatial.transform import Rotation as R


class BasePose:
    def __call__(self, bs) -> np.ndarray:
        raise NotImplementedError

    @property
    def canonical(self):
        raise NotImplementedError

    @property
    def canonical_vec(self):
        raise NotImplementedError

    @staticmethod
    def pose_to_vec_repr(pose: np.ndarray) -> np.ndarray:
        return pose[..., :2, :3].flatten(-2, -1)

    @property
    def repr_dim(self):
        return 6


class Sphere(BasePose):
    def __init__(self):
        self._canonical_vec = np.array([0, 0, 1.])
        # self._canonical = mat_33_to_44_np(canonical)
        # assert np.linalg.det(self._canonical) == 1, np.linalg.det(self._canonical)

    def _vec_forward_box(self, rot):
        # rot: (..., 3, 3)
        return np.einsum('...ij,j->...i', rot, self._canonical_vec)

    def _accept_fn(self, rot):
        vec = self._vec_forward_box(rot)
        return (vec[..., 2] < 0) & (-0.7 < vec[..., 1]) & (vec[..., 1] < 0.3)

    def __call__(self, bs):
        rot = rejection_sample_rot(bs, accept_fn=self._accept_fn)
        trans = self._vec_forward_box(rot)
        mat = assemble_rot_trans_np(rot, trans)

        rotvec = self._canonical_vec
        rotvec = rotvec * np.random.uniform(low=0, high=np.pi * 2, size=(bs,) if bs is not None else ())[
            ..., None]
        rot_roll = R.from_rotvec(rotvec=rotvec).as_matrix()
        mat = mat @ mat_33_to_44_np(rot_roll)
        return mat

    @property
    def canonical(self):
        raise NotImplementedError()

    @property
    def canonical_vec(self):
        return self._canonical_vec


SphereDown0p3 = Sphere


class Plane(BasePose):
    def __init__(self, cam_loc, rot_degree_range_scale, xy_range_scale, rot_roll_degree_range_scale):
        self.sample_fn = build_darkroom_rot_z_trans_plane_with_rot_roll(
            cam_loc=cam_loc,
            rot_degree_range_scale=rot_degree_range_scale,
            xy_range_scale=xy_range_scale,
            rot_roll_degree_range_scale=rot_roll_degree_range_scale,
        )
        # canonical pose
        c2p = get_lookat_mat(tuple(cam_loc))
        c2p_rot = c2p[:3, :3].numpy()
        p2c_rot = c2p_rot.T
        self._canonical = mat_33_to_44_np(p2c_rot)  # canonical to box
        self._canonical_vec = np.asarray([0, -1, 0])

    def __call__(self, bs) -> np.ndarray:
        return self.sample_fn(bs)

    @property
    def canonical(self):
        return self._canonical

    @property
    def canonical_vec(self):
        return self._canonical_vec


class Hemisphere(BasePose):
    def __init__(self):
        super().__init__()
        self.c2w_canonical = look_at(eye=(0, 4.0, -0.5)).numpy()
        self._canonical_vec = np.array([0, 0, 1.])

    @property
    def canonical(self):
        return mat_33_to_44_np(self.c2w_canonical.T)

    @property
    def canonical_vec(self):
        return self._canonical_vec

    def __call__(self, bs):
        rot = np.random.uniform(0, 1, size=(bs, 3) if bs is not None else (3,)) * (1, 0, 2 * np.pi)
        rot[..., 0] = np.abs(np.arccos(1 - 2 * rot[..., 0]) - np.pi / 2)
        rot = R.from_euler('XYZ', rot).as_matrix()

        rot = np.swapaxes(rot, -2, -1)
        c2w = rot @ self.c2w_canonical
        b2w = np.swapaxes(c2w, -2, -1)
        return mat_33_to_44_np(b2w)


### util functions ###

def rejection_sample_rot(bs: Union[int, None], accept_fn, max_tries=100) -> np.ndarray:
    r_all = np.zeros((0, 3, 3))
    bs_equiv = bs if bs is not None else 1
    while r_all.shape[0] < bs_equiv:
        r = R.random(bs_equiv * 10).as_matrix()  # (b', 3, 3)
        r = r[accept_fn(r), :, :]
        r_all = np.concatenate([r_all, r], axis=0)
        max_tries -= 1
        if max_tries == 0:
            print('infinite loop')
            import ipdb;
            ipdb.set_trace()

    if bs is None:
        r_all = r_all[0, :, :]
    else:
        r_all = r_all[:bs, :, :]
    return r_all


def uniform_sample_from_set_of_range(
        bs,
        spec: List[List],
        convert_degree_to_rad: bool,
):
    rand_raw = np.random.uniform(
        low=0, high=1, size=(bs,) if bs is not None else ())
    bins = np.asarray([r[1] - r[0] for r in spec])
    bin_starts = np.asarray([r[0] for r in spec])
    if convert_degree_to_rad:
        bins = bins / 180 * np.pi
        bin_starts = bin_starts / 180 * np.pi
    assert bins.sum() > 0, (bins, sum(bins))
    probs = np.cumsum([bin / bins.sum() for bin in bins])
    rot_indices = np.digitize(rand_raw, probs)
    rot = bin_starts[rot_indices] + rand_raw * bins[rot_indices]
    return rot

def build_rot_z_trans_plane(cam_loc: List, rot_degree_range_scale, xy_range_scale, vec_phy):
    # three coordinate frames: physical world, model camera, model world
    c2p = get_lookat_mat(tuple(cam_loc))  # camera to world_phy

    c2p_rot = c2p[:3, :3].numpy()
    p2c_rot = c2p_rot.T

    if isinstance(vec_phy, (tuple, list)):
       vec_phy = np.asarray(vec_phy)  # vertical to the floor in world_phy frame
    vec_cam = p2c_rot @ vec_phy  # vertical to the floor in camera frame
    # assume that the camera to world_model transformation has identity rotation
    # then the directional vector vec_cam is also in model world frame
    if isinstance(xy_range_scale, float) or isinstance(xy_range_scale, int):
        # legacy
        xy_range_scale = (xy_range_scale, xy_range_scale)
    xy_range_scale = tuple(xy_range_scale)
    if vec_cam[2] == 0:
        if xy_range_scale == (0, 0):
            pass
        elif vec_cam[0] == 0 and xy_range_scale[1] == 0:
            pass
        else:
            import ipdb; ipdb.set_trace()

    def fn(bs, rand_raw=None):
        rand_shape = (bs, 3) if bs is not None else (3,)
        if rand_raw is None:
            rand_raw = np.random.uniform(size=rand_shape)
        else:
            rand_raw = np.asarray(rand_raw)
            if rand_raw.shape != rand_shape:
                import ipdb; ipdb.set_trace()
                assert rand_raw.shape == (bs,) if bs is not None else (), rand_raw.shape
                rand_raw = np.ones(rand_shape) * rand_raw[..., None]

        # scale is in degree
        # when scale is a scalar, rotation will be [-scale/2, scale/2]
        # when scale is a list of list, e.g. [[-60, 120], [60, 120]],
        # rotation will be uniformly sampled from the union of the two intervals
        if isinstance(rot_degree_range_scale, (tuple, list)):
            bins = np.asarray([r[1] - r[0] for r in rot_degree_range_scale]) / 180 * np.pi
            bin_starts = np.asarray([r[0] for r in rot_degree_range_scale]) / 180 * np.pi
            assert bins.sum() > 0, (bins, sum(bins))
            probs = np.cumsum([bin / bins.sum() for bin in bins])
            rot_indices = np.digitize(rand_raw[..., 0], probs)
            rot = bin_starts[rot_indices] + rand_raw[..., 0] * bins[rot_indices]
        else:
            rot = (rand_raw[..., 0] - 0.5) * rot_degree_range_scale / 180 * np.pi
        if bs is None:
            rotvec = vec_phy * rot
        else:
            rotvec = vec_phy[None, :] * rot[:, None]
        rot = R.from_rotvec(rotvec).as_matrix()  # physical frame
        rot = p2c_rot @ rot  # camera frame

        # x, y, z are in model world frame
        # they live on a plane perpendicular to vec_cam
        # sample x, y such that the point is visible when projected to the image plane
        # the projection is done by taking x, y coordinates directly
        x = (rand_raw[..., 1] * 2 - 1) * xy_range_scale[0]
        y = (rand_raw[..., 2] * 2 - 1) * xy_range_scale[1]
        z = -(vec_cam[0] * x + vec_cam[1] * y) #/ vec_cam[2]
        if np.allclose(z, np.zeros_like(z)):
            z = np.zeros_like(z)
        else:
            assert vec_cam[2] != 0, vec_cam
            z = z / vec_cam[2]
        coord_cam = np.stack([x, y, z], axis=-1)

        mat = assemble_rot_trans_np(rot, coord_cam)
        return mat

    return fn


def build_darkroom_rot_z_trans_plane(cam_loc: List, rot_degree_range_scale, xy_range_scale):
    return build_rot_z_trans_plane(
        cam_loc, rot_degree_range_scale, xy_range_scale, vec_phy=(0, -1, 0))


def build_darkroom_rot_z_trans_plane_with_rot_roll(
        cam_loc: List, rot_degree_range_scale, xy_range_scale,
        rot_roll_degree_range_scale,
):
    fn = build_darkroom_rot_z_trans_plane(
        cam_loc=cam_loc, rot_degree_range_scale=rot_degree_range_scale, xy_range_scale=xy_range_scale)
    if isinstance(rot_roll_degree_range_scale, (list, tuple)):
        def fn2(bs, rand_raw=None):
            mat = fn(bs, rand_raw=rand_raw)
            rot = uniform_sample_from_set_of_range(bs=bs, spec=rot_roll_degree_range_scale, convert_degree_to_rad=True)
            rotvec = np.asarray([0, 0, 1.]) * rot[..., None]
            rot_roll = R.from_rotvec(rotvec).as_matrix()
            return mat @ mat_33_to_44_np(rot_roll)
        return fn2

    def fn2(bs, rand_raw=None):
        mat = fn(bs, rand_raw=rand_raw)
        rot_roll = R.from_rotvec(
            np.asarray([0., 0, 1])[None] * np.random.uniform(low=0, high=rot_roll_degree_range_scale / 180 * np.pi, size=bs if bs is not None else 1)[:,
                                           None]).as_matrix()
        if bs is None:
            rot_roll = rot_roll.squeeze(0)
        return mat @ mat_33_to_44_np(rot_roll)
    return fn2

