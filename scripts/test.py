import numpy as np
import argparse
import torch
import glob
import json
from tu.train_setup import set_seed
import os
from pathlib import Path
import src.models.generator
from src.utils.slerp import get_interpfn
import scipy.interpolate
from src.utils.pose import look_at, mat_33_to_44_np
from scipy.spatial.transform import Rotation as R
from src.utils.pose import assemble_rot_trans, get_tip_from_spherical_coord
import logging
from src.utils.test import run


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_canonical(cfg, model):
    if 'rose' in cfg['data']['kwargs']['dataset_folder']:
        p = get_tip_from_spherical_coord(elev=np.pi / 4, azim=0)
        mat_base = look_at(eye=(0, 0, 0), center=torch.tensor(p, dtype=torch.float32)).numpy()
        rotvec = model.generator.pose_prior.canonical_vec
        rotvec = rotvec * np.pi / 2 * 3
        rot_roll = R.from_rotvec(rotvec=rotvec).as_matrix()
        mat_base = mat_base @ rot_roll
        mat_base = mat_33_to_44_np(mat_base)
    elif 'tulip' in cfg['data']['kwargs']['dataset_folder']:
        p = get_tip_from_spherical_coord(elev=np.pi / 6, azim=0)
        mat_base = look_at(eye=(0, 0, 0), center=torch.tensor(p, dtype=torch.float32)).numpy()
        mat_base = mat_33_to_44_np(mat_base)
    elif 'green_crane' in cfg['data']['kwargs']['dataset_folder']:
        mat_base = model.generator.pose_prior.canonical
        p = get_tip_from_spherical_coord(elev=np.pi / 3, azim=0)
        p = look_at(eye=torch.tensor(p, dtype=torch.float32)).numpy()
        mat_base = mat_base @ mat_33_to_44_np(p)
    else:
        mat_base = model.generator.pose_prior.canonical
    return mat_base


def get_data_iter_latent_walk(cfg, model):
    # lerpv = get_interpfn(spherical=True, gaussian=True)
    lerpv = get_interpfn(spherical=True, gaussian=False)
    # lerpv = get_interpfn(spherical=False, gaussian=True)
    # lerpv = get_interpfn(spherical=False, gaussian=False)
    num_keyframes = 4
    z = model.generator.sample_latent(num_keyframes, {})['z']
    # with torch.no_grad():
    #     w = model.generator.renderer.sdf_network.style(z)
    # z[-1] = z[0]

    num_frames = 128
    cols = num_frames
    u_list = np.zeros((1, cols, z.shape[-1]))
    space = num_frames // num_keyframes
    anchors = z.cpu().numpy()
    # anchors = w.cpu().numpy()
    # compute anchors
    y = 0
    cur_anchor = 0
    for x in range(cols):
        if x % space == 0:
            u_list[y, x, :] = anchors[cur_anchor]
            cur_anchor = cur_anchor + 1
    # interpolate
    for x in range(cols):
        if x % space != 0:
            lastX = space * (x // space)
            nextX = lastX + space
            nextX = nextX % cols  # FIXME?
            fracX = (x - lastX) / float(space)
            u_list[y, x, :] = lerpv(fracX, u_list[y, lastX, :], u_list[y, nextX, :])
    u_list = torch.tensor(u_list, dtype=torch.float32, device='cuda')

    partial_data = {'b2w': torch.tensor(get_canonical(cfg, model), dtype=torch.float32, device='cuda')[None]}
    for x in range(cols):
        yield {
            'z': u_list[y, x, :][None],
            # 'z': torch.zeros_like(z[0])[None],
            # 'w': u_list[y, x, :][None],
            **partial_data,
        }


def get_data_iter_latent_walk(cfg, model):
    num_keyframes = 16
    z = model.generator.sample_latent(num_keyframes, {})['z']
    z[-1] = z[0]
    with torch.no_grad():
        w = model.generator.renderer.sdf_network.style(z)
    x = np.linspace(-0.0, 1.0, num_keyframes)
    y = w.cpu().numpy()
    interp = scipy.interpolate.interp1d(x, y, kind='cubic', axis=0)

    def get_z(ratio):
        return {
            'z': torch.zeros_like(z[0])[None],
            'w': torch.tensor(interp(ratio), dtype=torch.float32, device='cuda')[None],
        }
    # partial_data = model.generator.sample_prior(1, data={})
    partial_data = {'b2w': torch.tensor(get_canonical(cfg, model), dtype=torch.float32, device='cuda')[None]}
    n_frames = 256 if os.getenv('DEBUG') != '1' else 32
    for i in range(n_frames):
        yield {
            **get_z(i / n_frames),
            **partial_data,
        }


def get_rotation_matrix(tx, ty, tz):
    m_x = torch.zeros((len(tx), 3, 3)).to(tx.device)
    m_y = torch.zeros((len(tx), 3, 3)).to(tx.device)
    m_z = torch.zeros((len(tx), 3, 3)).to(tx.device)

    m_x[:, 1, 1], m_x[:, 1, 2] = tx.cos(), -tx.sin()
    m_x[:, 2, 1], m_x[:, 2, 2] = tx.sin(), tx.cos()
    m_x[:, 0, 0] = 1

    m_y[:, 0, 0], m_y[:, 0, 2] = ty.cos(), ty.sin()
    m_y[:, 2, 0], m_y[:, 2, 2] = -ty.sin(), ty.cos()
    m_y[:, 1, 1] = 1

    m_z[:, 0, 0], m_z[:, 0, 1] = tz.cos(), -tz.sin()
    m_z[:, 1, 0], m_z[:, 1, 1] = tz.sin(), tz.cos()
    m_z[:, 2, 2] = 1
    return torch.matmul(m_z, torch.matmul(m_y, m_x))


def get_transform_matrices(view):
    b = view.size(0)
    if view.size(1) == 6:
        rx = view[:,0]
        ry = view[:,1]
        rz = view[:,2]
        trans_xyz = view[:,3:].reshape(b,1,3)
    elif view.size(1) == 5:
        rx = view[:,0]
        ry = view[:,1]
        rz = view[:,2]
        delta_xy = view[:,3:].reshape(b,1,2)
        trans_xyz = torch.cat([delta_xy, torch.zeros(b,1,1).to(view.device)], 2)
    elif view.size(1) == 3:
        rx = view[:,0]
        ry = view[:,1]
        rz = view[:,2]
        trans_xyz = torch.zeros(b,1,3).to(view.device)
    rot_mat = get_rotation_matrix(rx, ry, rz)
    return rot_mat, trans_xyz


def get_data_iter_camera_walk(cfg, model):
    n_frames = 128 if os.getenv('DEBUG') != '1' else 16
    partial_data = model.generator.sample_latent(bs=1, data={})
    # model.generator.b2w_scene_prior = model.generator.pose_prior
    # mat_base = get_canonical(cfg, model)
    # rot = np.linspace(0, 360, n_frames, endpoint=False) * np.pi / 180
    # if 'cake' in cfg['data']['kwargs']['dataset_folder']:
    #     rot1 = np.linspace(-10, 10, n_frames // 2, endpoint=False) * np.pi / 180
    #     rot2 = np.linspace(10, -10, n_frames - n_frames // 2, endpoint=False) * np.pi / 180
    #     rot = np.concatenate([rot1, rot2])
    # rotvec = model.generator.pose_prior.canonical_vec  # (3,)
    # rotvec = rotvec * rot[..., None]  # (n_frames, 3)
    # rot_roll = R.from_rotvec(rotvec).as_matrix()
    # mat_all = mat_base @ mat_33_to_44_np(rot_roll)
    # mat_all = torch.tensor(mat_all, dtype=torch.float32, device='cuda')
    # mat_base = torch.tensor(model.generator.pose_prior.canonical, dtype=torch.float32, device='cuda')
    # mat_base = torch.eye(4, dtype=torch.float32, device='cuda')
    mat_base = torch.tensor(get_canonical(model), dtype=torch.float32, device='cuda')

    ## morph from target view to canonical
    morph_frames = 15
    view_zero = torch.tensor([0.15 * np.pi / 180 * 60, 0, 0, 0, 0, 0], device='cuda', dtype=torch.float32)
    morph_s = torch.linspace(0, 1, morph_frames, device='cuda')
    view_morph = morph_s.view(-1, 1, 1) * view_zero.view(1, 1, -1)# + (1 - morph_s.view(-1, 1, 1)) * self.view.unsqueeze(0)  # TxBx6

    ## yaw from canonical to both sides
    yaw_frames = 80
    # yaw_rotations = np.linspace(-np.pi / 2, 3 * np.pi / 2, yaw_frames)
    yaw_rotations = np.linspace(-np.pi, np.pi, yaw_frames)
    # yaw_rotations = np.concatenate([yaw_rotations[40:], yaw_rotations[::-1], yaw_rotations[:40]], 0)

    ## whole rotation sequence
    view_after = torch.cat([view_morph, view_zero.repeat(yaw_frames, 1, 1)], 0)  # TxBx6
    yaw_rotations = np.concatenate([np.zeros(morph_frames), yaw_rotations], 0)  # T

    def rearrange_frames(frames):
        # frames: (b=1, c, h, w)
        frames = torch.stack(frames, dim=1)
        morph_seq = frames[:, :morph_frames]
        yaw_seq = frames[:, morph_frames:]
        out_seq = torch.cat([
            morph_seq[:, :1].repeat(1, 5, 1, 1, 1),
            morph_seq,
            morph_seq[:, -1:].repeat(1, 5, 1, 1, 1),
            yaw_seq[:, yaw_frames // 2:],
            yaw_seq.flip(1),
            yaw_seq[:, :yaw_frames // 2],
            morph_seq[:, -1:].repeat(1, 5, 1, 1, 1),
            morph_seq.flip(1),
            morph_seq[:, :1].repeat(1, 5, 1, 1, 1),
        ], 1)
        out_seq = out_seq.unbind(1)
        return out_seq

    for i in range(yaw_rotations.shape[0]):
        ri = yaw_rotations[i]
        ri = torch.tensor([0, ri, 0], dtype=torch.float32, device='cuda').view(1, 3)
        rot_mat_i, _ = get_transform_matrices(ri)
        v_after_i = view_after[i]
        rot_mat, trans_xyz = get_transform_matrices(v_after_i)
        mat_i = assemble_rot_trans(rot_mat @ rot_mat_i, trans_xyz)
        # mat_i = mat_i @ mat_base
        mat_i = mat_base @ mat_i
        yield {
            **partial_data, 'b2w': mat_i,
        }
    yield rearrange_frames

    # for i in range(n_frames):
    #     yield {
    #         **partial_data,
    #         'b2w': mat_all[i:i+1, :, :],
    #     }


def get_data_iter_camera_walk(cfg, model):
    n_frames = 128 if os.getenv('DEBUG') != '1' else 16
    partial_data = model.generator.sample_latent(bs=1, data={})
    mat_base = get_canonical(cfg, model)
    rot = np.linspace(0, 360, n_frames, endpoint=False) * np.pi / 180
    rotvec = model.generator.pose_prior.canonical_vec  # (3,)
    rotvec = rotvec * rot[..., None]  # (n_frames, 3)
    rot_roll = R.from_rotvec(rotvec).as_matrix()
    mat_all = mat_base @ mat_33_to_44_np(rot_roll)
    mat_all = torch.tensor(mat_all, dtype=torch.float32, device='cuda')
    for i in range(n_frames):
        yield {
            **partial_data, 'b2w': mat_all[i:i+1],
        }


def main():
    main_fn(
        run_fn=run,
        get_data_iter_fn=get_data_iter_camera_walk,
        default_log_dir=f'logs/test_view')

    main_fn(
        run_fn=run, get_data_iter_fn=get_data_iter_latent_walk,
        default_log_dir=f'logs/test_latent')
    # main_fn(
    #     run_fn=run, get_data_iter_fn=get_data_iter_light_walk,
    #     default_log_dir=f'logs/test_light')
    # logger.info('Testing done.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', nargs='+', type=str, required=True, help='paths to evaluate')
    parser.add_argument('-t', '--tag', type=str, default='default', help='tag')
    parser.add_argument('-res', '--resolution', default=None, type=int, help='image crop resolution')
    parser.add_argument('-depth', '--depth-multiplier', default=None, type=int, help='multiplier for points sampled per ray')
    parser.add_argument('-ema', '--ema', type=float, default=None, help='EMA 0.9 or 0.99', choices=[0.9, 0.99, None])
    parser.add_argument('-dry', '--dry', action='store_true', help='dry run')
    parser.add_argument('--force-update', action='store_true', help='overwrite folder SLURM_ID/CHECKPOINT_NAME with rerendered results')
    return parser.parse_args()


def main_fn(run_fn, get_data_iter_fn, default_log_dir):
    args = parse_args()

    set_seed(0)
    if args.depth_multiplier is not None:
        src.models.generator.MAX_RAY_BATCH_SIZE /= args.depth_multiplier
    if args.resolution is not None:
        src.models.generator.MAX_RAY_BATCH_SIZE /= (args.resolution / 128) ** 2

    logger.info(f'found paths {len(args.path)}')
    paths = args.path
    paths = list(reversed(sorted(paths, key=os.path.getmtime)))
    logger.info(f'total runs {len(paths)}')

    html_dirs = []
    for path in paths:
        if os.path.isfile(path):
            out_dir = os.path.abspath(os.path.join(path, os.pardir, os.pardir))
            checkpoint_paths = [path]
        else:
            out_dir = path
            checkpoint_paths = glob.glob(os.path.join(out_dir, 'checkpoints', 'model.pt'))
        cfg_path = os.path.join(out_dir, 'cfg.json')
        if os.path.exists(cfg_path):
            with open(cfg_path, 'r') as f:
                cfg = json.load(f)
        else:
            cfg = {'runtime': {'slurm_job_id': 'dummy', 'slurm_job_name': 'dummy'}}

        checkpoint_paths = list(reversed(sorted(checkpoint_paths, key=os.path.getmtime)))

        for checkpoint_path in checkpoint_paths:
            if not args.dry:
                html_basename = Path(checkpoint_path).stem
                if args.ema is not None:
                    html_basename = f"{html_basename}_ema_{str(args.ema).split('.')[-1]}"
                html_dir = os.path.join(
                    default_log_dir, str(cfg['runtime']['slurm_job_id']), html_basename
                )
                html_dirs.append(html_dir)

                _, model = run_fn(
                    log_dir=checkpoint_path,
                    get_data_iter_fn=get_data_iter_fn, vi=html_dir,
                    test_resolution=args.resolution, depth_multiplier=args.depth_multiplier, ema=args.ema,
                    force_update=args.force_update,
                )


if __name__ == "__main__":
    main()
