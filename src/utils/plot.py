import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
from tu.common import vi_helper
from torchvision.transforms import ToTensor
import io


def fig2img(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def get_camera_wireframe(scale: float = 0.03):
    """
    Returns a wireframe of a 3D line-plot of a camera symbol.
    Shape (15, 4)
    """
    a = 0.5 * torch.tensor([-2, 1.5, 4])
    up1 = 0.5 * torch.tensor([0, 1.5, 4])
    up2 = 0.5 * torch.tensor([0, 2, 4])
    b = 0.5 * torch.tensor([2, 1.5, 4])
    c = 0.5 * torch.tensor([-2, -1.5, 4])
    d = 0.5 * torch.tensor([2, -1.5, 4])
    C = torch.zeros(3)
    F = torch.tensor([0, 0, 3])
    camera_points = [a, up1, up2, up1, b, d, c, a, C, b, d, C, c, C, F]
    lines = torch.stack([x.float() for x in camera_points]) * scale
    ones = torch.ones(lines.shape[0], 1)
    lines = torch.cat([lines, ones], dim=1)
    return lines


def plot_cameras(ax, c2w, color: str = "blue", scale=1.0):
    # c2w: (bs, 4, 4)
    device = c2w.device
    plot_handles = []
    cam_wires_canonical = get_camera_wireframe(scale).to(device)  # 15, 4
    cam_wires_trans = (c2w[:, None, :, :] @ cam_wires_canonical[None, :, :, None]).squeeze(-1)  # bs, 15, 4
    for wire in cam_wires_trans:
        x_, y_, z_, _ = wire.detach().cpu().numpy().T.astype(float)
        (h,) = ax.plot(x_, y_, z_, color=color, linewidth=0.3)
        plot_handles.append(h)
    return plot_handles


def plot_camera_scene(c2w, c2w_gt=None, plot_radius=None, return_fig=False, rel_scale=.05):
    if plot_radius is None:
        plot_radius = max(c2w[..., :3, 3].abs().max().item(), 1)
    assert plot_radius > 0, plot_radius
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=45., azim=60)
    ax.set_xlim3d([-plot_radius, plot_radius])
    ax.set_ylim3d([-plot_radius, plot_radius])
    #     ax.set_zlim3d([0, plot_radius * 2])
    ax.set_zlim3d([-plot_radius, plot_radius])

    xspan, yspan, zspan = 3 * [np.linspace(0, plot_radius, 20)]
    zero = np.zeros_like(xspan)
    ax.plot3D(xspan, zero, zero, 'k--')
    ax.plot3D(zero, yspan, zero, 'k--')
    #     ax.plot3D(zero, zero, zspan + plot_radius, 'k--')
    ax.plot3D(zero, zero, zspan, 'k--')
    # ax.text(plot_radius, .5, .5, "x", color='red')
    # ax.text(.5, plot_radius, .5, "y", color='green')
    # #     ax.text(.5, .5, plot_radius * 2, "z", color='blue')
    # ax.text(.5, .5, plot_radius, "z", color='blue')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    scale = rel_scale * plot_radius
    handle_cam = plot_cameras(ax, c2w, color="#FF7D1E", scale=scale)
    if c2w_gt is not None:
        handle_cam_gt = plot_cameras(ax, c2w_gt, color="#812CE5", scale=scale)

        labels_handles = {
            "Est": handle_cam[0],
            "GT": handle_cam_gt[0],
        }
    else:
        labels_handles = {
            # "Estimated cameras": handle_cam[0]
        }

    ax.legend(
        labels_handles.values(),
        labels_handles.keys(),
        loc="upper center",
        bbox_to_anchor=(0.32, 0.7),
        prop={'size': 8}
    )

    # ax.axis('off')
    fig.tight_layout()
    if return_fig:
        return fig, ax

    img = fig2img(fig)

    plt.close(fig)

    img = ToTensor()(img)

    return img


def plot_hist(arr, vi, hist_kwargs=None):
    fig, ax = plt.subplots(1, 1)
    if hist_kwargs is None:
        hist_kwargs = dict()
    _ = ax.hist(arr, **hist_kwargs)  # arguments are passed to np.histogram
    vi_helper.dump_table(vi, [[fig]], col_names=['histogram'])
    plt.close(fig)


def normalize_batched_tensor(x, q=0., xmin=None, xmax=None):
    # similar to normalize_tensor,
    # but xmin and xmax are independent for elements in the batch
    # assume batch dim = 0
    assert q < .5, q
    if xmin is None:
        xmin = torch.quantile(x.flatten(1), q, dim=-1)
        xmin = xmin.reshape((x.shape[0],) + (1,) * (x.ndim - 1))
    if xmax is None:
        xmax = torch.quantile(x.flatten(1), 1 - q, dim=-1)
        xmax = xmax.reshape((x.shape[0],) + (1,) * (x.ndim - 1))
    return ((x - xmin) / (xmax - xmin)).clamp(0, 1)
