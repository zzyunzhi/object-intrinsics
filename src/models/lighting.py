import torch.nn as nn
import torch.nn.functional as F
import torch


class DirectionalLightWithSpecularFixInit(nn.Module):
    def __init__(
            self, direction,
            ambient_color: float = 0.33,
            diffuse_color: float = 0.66,
            specular_color: float = 0.01,
            shininess: float = 10,
    ) -> None:
        super().__init__()
        init_ambient_ratio = ambient_color / (ambient_color + diffuse_color)
        init_ambient = get_tensor(init_ambient_ratio).logit_()
        self.register_parameter('param_ambient', nn.Parameter(init_ambient))
        self.register_parameter('param_direction', nn.Parameter(get_tensor(direction)))
        assert torch.allclose(torch.linalg.norm(self.direction), torch.ones(())), (self.direction, torch.linalg.norm(self.direction))
        self.register_parameter('param_shininess', nn.Parameter(get_tensor(shininess)))
        self.register_parameter('param_specular', nn.Parameter(get_tensor(specular_color)))

        assert self.param_direction.shape == (3,), self.param_direction
        assert self.param_ambient.shape == (), self.param_ambient
        assert self.param_shininess.shape == (), self.param_shininess
        assert self.param_specular.shape == (), self.param_specular
        # logging.info(str(self))

    def __repr__(self):
        return f"DirectionalLightWithSpecularFixInit(" \
               f"ambient_color={self.ambient_color}, " \
               f"diffuse_color={self.diffuse_color}, " \
               f"specular_color={self.specular_color}, " \
               f"shininess={self.shininess}, " \
               f"direction={self.direction}, " \
               f"), optimize {[n for n, _ in self.named_parameters()]}"

    @property
    def specular_color(self):
        return self.param_specular.expand(3).clamp(min=0)

    @property
    def ambient_color(self):
        return torch.sigmoid(self.param_ambient).expand(3)

    @property
    def diffuse_color(self):
        return (1 - torch.sigmoid(self.param_ambient)).expand(3)

    @property
    def shininess(self):
        return self.param_shininess

    @property
    def direction(self):
        return self.param_direction / torch.linalg.norm(self.param_direction)

    def batch_transform(self, *, w2b: torch.Tensor):
        return BatchDirectionalLightWithSpecularFixInit(self, w2b)

    def diffuse(self, normals, points=None) -> torch.Tensor:
        return diffuse(
            normals=normals,
            color=self.diffuse_color,
            direction=self.direction,
        )

    def specular(self, normals, camera_position, points) -> torch.Tensor:
        return specular(
            normals=normals,
            color=self.specular_color,
            direction=self.direction,
            camera_position=camera_position,
            points=points,
            shininess=self.shininess,
        )


class BatchDirectionalLightWithSpecularFixInit:
    def __init__(self, light: DirectionalLightWithSpecularFixInit, w2b: torch.Tensor):
        # w2b: SE(3) (b, 4, 4)
        assert w2b.shape[1:] == (4, 4), w2b.shape
        self.light = light
        self.w2b = w2b

    @property
    def ambient_color(self):
        return self.light.ambient_color

    @property
    def batch_size(self):
        return self.w2b.shape[0]

    def specular(self, normals, camera_position, points) -> torch.Tensor:
        assert normals.shape[0] == self.batch_size and normals.shape[2] == 3 and len(normals.shape) == 3, normals.shape
        return specular(
            normals=normals,
            color=self.light.specular_color,
            direction=self.batch_direction(points),
            camera_position=camera_position,
            points=points,
            shininess=self.light.shininess,
        )

    def diffuse(self, normals, points=None):
        # normals, points: shape (b, n_pts, 3)
        assert normals.shape[0] == self.batch_size and normals.shape[2] == 3 and len(normals.shape) == 3, normals.shape
        return diffuse(
            normals=normals,
            color=self.light.diffuse_color,
            # direction=torch.einsum('bij,j->bi', self.w2b[:, :3, :3], self.light.direction)[:, None, :]
            direction=self.batch_direction(points)
        )

    def batch_direction(self, points):
        # points: (bs, n_pts, 3)
        assert points.shape[0] == self.batch_size and points.shape[2] == 3 and len(points.shape) == 3, points.shape
        direction = torch.einsum('bij,j->bi', self.w2b[:, :3, :3], self.light.direction)  # (bs, 3)
        return direction[:, None, :].expand_as(points)  # (bs, n_pts, 3)


# https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/renderer/lighting.html#PointLights
# https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/renderer/mesh/shading.html?highlight=lighting


def diffuse(normals, color, direction) -> torch.Tensor:
    """
    Calculate the diffuse component of light reflection using Lambert's
    cosine law.

    Args:
        normals: (N, ..., 3) xyz normal vectors. Normals and points are
            expected to have the same shape.
        color: (1, 3) or (N, 3) RGB color of the diffuse component of the light.
        direction: (x,y,z) direction of the light

    Returns:
        colors: (N, ..., 3), same shape as the input points.

    The normals and light direction should be in the same coordinate frame
    i.e. if the points have been transformed from world -> view space then
    the normals and direction should also be in view space.

    NOTE: to use with the packed vertices (i.e. no batch dimension) reformat the
    inputs in the following way.

    .. code-block:: python

        Args:
            normals: (P, 3)
            color: (N, 3)[batch_idx, :] -> (P, 3)
            direction: (N, 3)[batch_idx, :] -> (P, 3)

        Returns:
            colors: (P, 3)

        where batch_idx is of shape (P). For meshes, batch_idx can be:
        meshes.verts_packed_to_mesh_idx() or meshes.faces_packed_to_mesh_idx()
        depending on whether points refers to the vertex coordinates or
        average/interpolated face coordinates.
    """
    # Reshape direction and color so they have all the arbitrary intermediate
    # dimensions as normals. Assume first dim = batch dim and last dim = 3.

    # Renormalize the normals in case they have been interpolated.
    # We tried to replace the following with F.cosine_similarity, but it wasn't faster.
    normals = F.normalize(normals, p=2, dim=-1, eps=1e-6)
    direction = F.normalize(direction, p=2, dim=-1, eps=1e-6)
    angle = F.relu(torch.sum(normals * direction, dim=-1))
    return color * angle[..., None]


def specular(
        points, normals, direction, color, camera_position, shininess
) -> torch.Tensor:
    """
    Calculate the specular component of light reflection.
    Args:
        points: (N, ..., 3) xyz coordinates of the points.
        normals: (N, ..., 3) xyz normal vectors for each point.
        color: (N, 3) RGB color of the specular component of the light.
        direction: (N, 3) vector direction of the light.
        camera_position: (N, 3) The xyz position of the camera.
        shininess: (N)  The specular exponent of the material.
    Returns:
        colors: (N, ..., 3), same shape as the input points.
    The points, normals, camera_position, and direction should be in the same
    coordinate frame i.e. if the points have been transformed from
    world -> view space then the normals, camera_position, and light direction
    should also be in view space.
    To use with a batch of packed points reindex in the following way.
    .. code-block:: python::
        Args:
            points: (P, 3)
            normals: (P, 3)
            color: (N, 3)[batch_idx] -> (P, 3)
            direction: (N, 3)[batch_idx] -> (P, 3)
            camera_position: (N, 3)[batch_idx] -> (P, 3)
            shininess: (N)[batch_idx] -> (P)
        Returns:
            colors: (P, 3)
        where batch_idx is of shape (P). For meshes batch_idx can be:
        meshes.verts_packed_to_mesh_idx() or meshes.faces_packed_to_mesh_idx().
    """
    assert points.shape == normals.shape, (points.shape, normals.shape)
    assert points.shape == direction.shape, (points.shape, direction.shape)
    assert camera_position.shape == normals.shape, (camera_position.shape, normals.shape)
    assert shininess.shape == (), shininess.shape
    # Renormalize the normals in case they have been interpolated.
    # We tried a version that uses F.cosine_similarity instead of renormalizing,
    # but it was slower.
    normals = F.normalize(normals, p=2, dim=-1, eps=1e-6)
    direction = F.normalize(direction, p=2, dim=-1, eps=1e-6)
    cos_angle = torch.sum(normals * direction, dim=-1)
    # No specular highlights if angle is less than 0.
    mask = (cos_angle > 0).to(torch.float32)

    # Calculate the specular reflection.
    view_direction = camera_position - points
    view_direction = F.normalize(view_direction, p=2, dim=-1, eps=1e-6)
    reflect_direction = -direction + 2 * (cos_angle[..., None] * normals)

    # Cosine of the angle between the reflected light ray and the viewer
    alpha = F.relu(torch.sum(view_direction * reflect_direction, dim=-1)) * mask
    return color * torch.pow(alpha, shininess)[..., None]


def get_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    return torch.tensor(x, dtype=torch.float32)
