from torch.nn.parallel import DistributedDataParallel
from intrinsics.utils.train import read_render_config
from torch.utils.data.distributed import DistributedSampler
import copy
import torch.distributed as dist
import torch
from pathlib import Path
import json
from torch.utils.tensorboard import SummaryWriter
import tu
from tu.utils.config import build_from_config, check_cfg_consistency
from tu.train_setup import count_parameters
import os
import logging
from tu.train.setup import get_cfg, get_parser
from tu.train.utils import overwrite_cfg_from_dotlist
from tu.train_setup import open_tensorboard, set_seed_benchmark
from tu.utils.config import overwrite_cfg


logger = logging.getLogger(__name__)


def setup_ddp():
    # torch.backends.cudnn.benchmark = True
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', init_method='env://')

    logger.info(f'setting up ddp: {dist.get_rank()} / {dist.get_world_size()}')


def get_dataloaders(cfg, args, use_ddp):
    # https://github.com/pytorch/elastic/blob/bc88e6982961d4117e53c4c8163ecf277f35c2c5/examples/imagenet/main.py#L268
    dataset = build_from_config(cfg['data'])
    # hack
    if hasattr(dataset, 'dataloader'):
        # FIXME should pass in split = train / val
        # FIXME will it work for use_ddp=True?
        return dataset.dataloader(), dataset.dataloader()

    # https://pytorch.org/docs/stable/data.html
    if use_ddp:
        train_sampler = DistributedSampler(dataset, seed=args.seed)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg['training']['batch_size'],  # batch size per gpu
        num_workers=2 if os.getenv('DEBUG') != '1' else 0, shuffle=True if train_sampler is None else False,
        pin_memory=True, drop_last=True,
        sampler=train_sampler,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg['training']['batch_size'],
        num_workers=2 if os.getenv('DEBUG') != '1' else 0, shuffle=False,
        pin_memory=True, drop_last=False,
    )
    return train_loader, val_loader


def overwrite_from_dataset(cfg):
    path = cfg['dataset']
    render_config = read_render_config(path, os.path.join(path, 'cfg.yaml'))

    if isinstance(render_config['scene_fov'], list):
        # FIXME hack
        overwrite_cfg(cfg['data_info'], 'scene_fov', render_config['scene_fov'][0])
    else:
        overwrite_cfg(cfg['data_info'], 'scene_fov', render_config['scene_fov'])

    overwrite_cfg(cfg['data_info'], 'fov', render_config['fov'])
    overwrite_cfg(cfg['data_info'], 'cam_dist', render_config['cam_dist'])
    overwrite_cfg(cfg['data_info'], 'pose_prior', render_config['b2w_scene_prior'])

    if 'l2w_scene_prior' in render_config:
        overwrite_cfg(cfg['data_info'], 'cam_loc', render_config['l2w_scene_prior']['cam_loc'])
        overwrite_cfg(cfg['data_info'], 'light_loc', render_config['l2w_scene_prior']['light_loc'])
    else:
        logger.warning('no l2w_scene_prior in render_config, use collocated light')

    if isinstance(render_config['img_size_scene'], list):
        # FIXME hack
        scene_res = int(cfg['resolution'] * render_config['img_size_scene'][0] / render_config['img_size'])
    else:
        scene_res = int(cfg['resolution'] * render_config['img_size_scene'] / render_config['img_size'])
    overwrite_cfg(cfg, 'scene_resolution', scene_res)
    if 'opts' in render_config:
        overwrite_cfg_from_dotlist(cfg, render_config['opts'])


def main():
    if int(os.getenv('WORLD_SIZE', 0)) > 1:
        setup_ddp()
        # some code depends on dist.is_initialized()
        rank = dist.get_rank()
    else:
        rank = 0
    if os.getenv('DEBUG') == '1':
        torch.autograd.set_detect_anomaly(True)

    parser = get_parser()
    parser.set_defaults(config='train')
    parser._optionals._option_string_actions['--config'].required = False

    args = parser.parse_args()

    logger.info(json.dumps(vars(args)))

    set_seed_benchmark(args.seed + rank)

    cfg = get_cfg(args, overwrite_fn_before_resolve=lambda cfg_: overwrite_from_dataset(cfg_))

    if os.environ.get('SLURM_JOB_NAME') == 'bash' or os.getenv('DEBUG') == '1':
        overwrite_cfg(cfg['training']['train_loops_fn']['kwargs'], 'visualize_every', 100)
        overwrite_cfg(cfg['training']['train_loops_fn']['kwargs'], 'print_every', 100)
        # overwrite_cfg(cfg['training']['train_loops_fn']['kwargs'], 'checkpoint_every', 99999)
        # overwrite_cfg(cfg['training']['train_loops_fn']['kwargs'], 'eval_every', 100)

    train_loader, val_loader = get_dataloaders(cfg, args, use_ddp=dist.is_initialized())

    modules = dict()
    def get_model(name, key=None, **kwargs):
        key = name if key is None else key
        model = build_from_config(cfg['model'][key], **kwargs)
        model = model.cuda(rank)
        # need broadcast_buffers=False
        # because register_buffer from ADA augment.py
        # will error in upfirdn2d.py
        # alternatively can use self.Hz_geom.clone() instead of self.Hz_geom
        if dist.is_initialized():
            model = DistributedDataParallel(model, device_ids=[rank], broadcast_buffers=True if 'discriminator' not in name else False)
        if 'params' in cfg['training'][f'opt_{name}']['kwargs']:
            opt = copy.deepcopy(cfg['training'][f'opt_{name}'])
            param_groups = opt['kwargs']['params']
            params = []
            other_params = []
            for k in param_groups:
                child = getattr(model, k)
                params.append({'name': k, 'params': child.parameters(), **param_groups[k]})
            for k, child in model.named_children():
                if k in param_groups:
                    continue
                other_params.extend(child.parameters())
            params.append({'name': 'default', 'params': other_params})
            opt['kwargs']['params'] = params
            opt = build_from_config(opt)
        else:
            opt = build_from_config(cfg['training'][f'opt_{name}'], params=model.parameters())
        sch = build_from_config(cfg['training'][f'sch_{name}'], optimizer=opt)
        modules[name] = model
        modules[f'opt_{name}'] = opt
        modules[f'sch_{name}'] = sch

    for key in ['generator', 'discriminator', 'mask_discriminator']:
        get_model(key)
        logger.info(f"{key} params {count_parameters(modules[key], verbose=False)}")

    log_dir = cfg['log_dir']
    if rank == 0:
        writer = SummaryWriter(log_dir)
        open_tensorboard(log_dir)
        logger.info(f'tensorboard --bind_all --logdir {Path(log_dir).absolute()}')
    else:
        writer = None

    trainer = build_from_config(cfg['trainer'], modules=modules, writer=writer)

    epoch = load_checkpoint(trainer, cfg, cfg['training']['checkpoint_dir'])

    build_from_config(cfg['training']['train_loops_fn'], cfg=cfg,
                      trainer=trainer, train_loader=train_loader, val_loader=val_loader, epoch=epoch)


def load_checkpoint(trainer, cfg, path, strict=False):
    if path is None:
        return -1

    with open(os.path.abspath(os.path.join(path, '../../cfg.json')), 'r') as f:
        checkpoint_cfg = json.load(f)
    _ = check_cfg_consistency(cfg, checkpoint_cfg, ignore_keys={'log_dir', 'runtime.*', 'training.*', 'trainer.*'})
    unused = trainer.load_checkpoint(path, strict=strict)
    return unused['epoch']


if __name__ == "__main__":
    main()
