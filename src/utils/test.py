from typing import Callable
from tu.configs import list_of_dicts__to__dict_of_lists
from tu.loggers.utils import setup_vi
from tu.utils.visualize import to_frames, get_image
from tu.utils.config import overwrite_cfg
import torch
from pathlib import Path
import json
from .checkpoint import CheckpointIO
from tu.common import vi_helper
from tu.loggers.utils import print_viscam_url  # FIXME when release
import logging
from tu.utils.config import build_from_config
import os
from tu.configs import AttrDict


logger = logging.getLogger(__name__)


def run(log_dir: str, get_data_iter_fn: Callable, vi: str, save_frames=False,
        test_resolution=None, depth_multiplier=None, ema=None, force_update=False):

    cfg, model, model_info = inference_setup(log_dir, test_resolution, depth_multiplier, ema=ema)

    html_dir = vi
    if not force_update and any(Path(html_dir).glob('assets/*/*.mp4')):
        logger.info(f'found existing path: {html_dir}')
        return cfg, model
    vi, _ = setup_vi(html_dir, exist_ok=True)

    if model is None:
        return None, None

    model = AttrDict(vi_helper=vi_helper, vi=vi, **model)
    data_iter = iter(get_data_iter_fn(cfg, model))
    run_for_model(model, data_iter)

    if save_frames:
        raise NotImplementedError()

    vi_helper.dump_table(vi, [
        [print_viscam_url(cfg['log_dir'], verbose=False)],
        [cfg['runtime']['slurm_job_id']],
        [f"epoch {model_info['epoch']}, it {model_info['it']}, {model_info['checkpoint_path']}"],
    ], table_name='')

    return cfg, model


def update_legacy_config_for_inference(cfg):
    pass


def update_config(cfg, test_resolution=None, depth_multiplier=None):
    if depth_multiplier is not None:
        overwrite_cfg(cfg['model']['generator']['kwargs']['renderer']['kwargs'], 'n_importance',
                      cfg['model']['generator']['kwargs']['renderer']['kwargs']['n_importance'] * depth_multiplier)
        overwrite_cfg(cfg['model']['generator']['kwargs']['renderer']['kwargs'], 'n_samples',
                      cfg['model']['generator']['kwargs']['renderer']['kwargs']['n_samples'] * depth_multiplier)

    if test_resolution is not None:
        upsample_ratio = test_resolution / cfg['resolution']
        test_resolution = int(cfg['resolution'] * upsample_ratio)
        overwrite_cfg(cfg, 'resolution', test_resolution)
        overwrite_cfg(cfg['model']['generator']['kwargs'], 'resolution', test_resolution)


def update_legacy_state_dict(model, state_dict):
    # legacy state_dict did not save camera
    if next(iter(state_dict['generator'].keys())).startswith('module.'):
        prefix = 'module.'
    else:
        prefix = ''
    # if f'{prefix}camera.?' not in state_dict['generator']:
    #     for k, v in model.generator.camera.state_dict().items():
    #         state_dict['generator'][f'{prefix}camera.{k}'] = v


def load_state_dict(model, state_dict, strict=True):
    generator = model.generator
    checkpoint_io = CheckpointIO('/viscam/u/yzzhang/tmp', generator=generator)
    load_dict = checkpoint_io.load(state_dict, strict=strict)
    logger.info(f"epoch {load_dict['epoch']}, it {load_dict['it']}")
    return load_dict


def inference_setup(log_dir, test_resolution=None, depth_multiplier=None, ema=None, load_model=True):
    log_dir = Path(log_dir)
    if log_dir.is_file():
        path_to_cfg = log_dir.parent.parent / 'cfg.json'
        path_to_checkpoint = log_dir
    else:
        path_to_cfg = log_dir / 'cfg.json'
        path_to_checkpoint = log_dir / 'checkpoints' / 'model.pt'

    with open(path_to_cfg, 'r') as f:
        cfg = json.load(f)
    update_legacy_config_for_inference(cfg)
    update_config(cfg, test_resolution=test_resolution, depth_multiplier=depth_multiplier)
    if not load_model:
        return cfg, None, None

    if not os.path.exists(path_to_checkpoint):
        logger.error(f'checkpoint not found: {path_to_checkpoint}')
        return cfg, None, None

    logger.info(f'loading from {path_to_checkpoint}')
    state_dict = torch.load(path_to_checkpoint)
    if state_dict['it'] == 0:
        return cfg, None, None

    if ema is not None:
        state_dict = dict(generator=state_dict[f'ema@{ema}']['state_dict'], epoch=state_dict['epoch'], it=state_dict['it'])

    generator = build_from_config(cfg['model']['generator']).cuda()

    generator.bg_color = lambda bs: torch.ones((), dtype=torch.float32) # load_bg_color_fn('white', generator.resolution, generator.resolution)

    _ = generator.eval()
    for p in generator.parameters():
        p.requires_grad = False
    model = AttrDict(generator=generator)
    update_legacy_state_dict(model, state_dict)
    load_dict = load_state_dict(model, state_dict, strict=False)

    model_info = {**load_dict, 'checkpoint_path': path_to_checkpoint}
    return cfg, model, model_info


def run_for_model(model, data_iter):
    rearrange_fn = None
    maps_all = []
    for data in data_iter:
        if not isinstance(data, dict):
            rearrange_fn = data  # hack
            continue
        with torch.no_grad():
            blob = model.generator(bs=1, data=data, it=None, return_raw=True)

        blob = blob['box']
        maps = {
            'image': blob['render_out']['image'],
            'normal': blob['render_out']['normal_map'].flip(-3) * 0.5 + 0.5,
            'shading': blob['render_out']['shading_map'],
        }
        maps_all.append(maps)

    maps_all = list_of_dicts__to__dict_of_lists(maps_all)
    if rearrange_fn is not None:
        maps_all = {k: rearrange_fn(maps_all[k]) for k in maps_all.keys()}
    maps_all = {k: list(map(get_image, maps_all[k])) for k in maps_all.keys()}

    model.vi_helper.dump_table(model.vi, [[to_frames(maps_all[k]) for k in maps_all.keys()]],
                               col_names=list(maps_all.keys()), table_name='generator')
