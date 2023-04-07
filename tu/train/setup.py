import datetime
import os
import sys

import yaml
import json
import logging
from typing import Dict, Tuple
from .utils import overwrite_cfg_from_dotlist, overwrite_cfg, update_cfg_slurm, resolve_with_omegaconf
import torch.distributed as dist
import argparse


logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--overwrite', action='store_true', help='overwrite output dir')
    parser.add_argument('-s', '--seed', type=int, default=0, help='seed')
    parser.add_argument('-t', '--tag', default=None, type=str, help='tag appended to output dir')
    parser.add_argument('-d', '--dataset', type=str, required=True, help='dataset directory')
    parser.add_argument('-c', '--config', type=str, required=True, help='config name or path to config yaml')
    parser.add_argument('--log-unique', action='store_true', help='append timestamp to logging dir')
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser


def debug_overwrite(cfg, args):
    if os.getenv('DEBUG') == '1':
        logger.info('DEBUG overwrite')
        args.tag = 'debug' if args.tag is None else args.tag + '_debug'


def get_log_dir(cfg, args):
    tag = f"_{args.dataset}".replace('/', '_')
    tag += '_' + args.config.split('/')[-1].removesuffix('.yaml')
    if args.tag is not None:
        tag += f'_{args.tag}'
    if os.environ.get('SLURM_JOB_NAME') == 'bash':
        tag += '_local'
    if os.environ.get('DEBUG') == '1':
        tag += '_debug'
    if args.log_unique:
        tag += f"_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = cfg['log_dir']
    if log_dir is not None:
        raise NotImplementedError(log_dir)
    log_dir = os.path.join('logs', tag)
    return log_dir


def get_cfg(args, overwrite_fn_before_resolve=None):
    # assume cfg has field cfg['log_dir'], cfg['dataset']
    # add fields cfg['runtime']
    if os.path.exists(args.config):
        cfg_path = args.config
    else:
        cfg_path = f'configs/{args.config}.yaml'
    cfg = load_cfg_from_path(cfg_path)
    overwrite_cfg(cfg, "dataset", args.dataset)
    debug_overwrite(cfg, args)

    if len(args.opts) > 0:
        assert args.tag is not None, f"specify tag for {args.opts}"
    log_dir = get_log_dir(cfg, args)
    if os.path.exists(log_dir) and not args.overwrite:
        logger.error(f'{log_dir} exists')
        exit(1)
    overwrite_cfg(cfg, 'log_dir', log_dir)
    
    if overwrite_fn_before_resolve is not None:
        overwrite_fn_before_resolve(cfg)

    overwrite_cfg_from_dotlist(cfg, args.opts)

    cfg = resolve_with_omegaconf(cfg)

    if dist.is_initialized() and dist.get_rank() != 0:
        return cfg

    os.makedirs(log_dir, exist_ok=args.overwrite)
    update_cfg_slurm(cfg, log_dir)
    overwrite_cfg(cfg['runtime'], 'args', vars(args), check_exists=False)
    overwrite_cfg(cfg['runtime'], 'argv', sys.argv, check_exists=False)

    with open(os.path.join(log_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)
    with open(os.path.join(log_dir, 'cfg.json'), 'w') as f:
        json.dump(cfg, f, sort_keys=True, indent=4)
    return cfg


def load_cfg_from_path(path, **kwargs) -> Dict:
    path = os.path.abspath(path)
    if path.endswith('.yaml'):
        with open(path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.Loader)
    elif path.endswith('.json'):
        with open(path, 'r') as f:
            cfg = json.load(f)
    else:
        raise NotImplementedError(path)
    cfg = load_cfg_from_dict(cfg, **kwargs)
    return cfg


def load_cfg_from_dict(cfg, allow_new_key=False, write_env_vars=True) -> Dict:
    if '__allow_new_key__' in cfg:
        allow_new_key = allow_new_key or cfg.pop('__allow_new_key__')
    if '_BASE_' in cfg:
        if cfg['_BASE_'] is not None and cfg['_BASE_'].startswith('$'):
            # temporarily resolve cfg to determine _BASE_
            cfg_base_path = resolve_with_omegaconf(cfg)['_BASE_']
        else:
            cfg_base_path = cfg['_BASE_']
        _ = cfg.pop('_BASE_')
        if cfg_base_path is not None:
            cfg_base = load_cfg_from_path(cfg_base_path, allow_new_key=allow_new_key, write_env_vars=False)  # recursive
            if 'runtime' in cfg_base:
                _ = cfg_base.pop('runtime')
            cfg, _ = update_recursive(cfg_base, cfg, allow_new_key=allow_new_key)

    if write_env_vars and '_ENV_VARS_' in cfg:
        # set to true only in the first call
        # otherwise env vars from base will be overwritten
        env_vars = cfg['_ENV_VARS_']
        for k, v in env_vars.items():
            v = str(v)
            logger.info(f'set environ: {k}: {os.getenv(k)} -> {v}')
            os.environ[k] = v
    return cfg


def update_recursive(dict1: Dict, dict2: Dict, allow_new_key=False) -> Tuple[Dict, Dict]:
    """

    :param dict1: muted
    :param dict2:
    :param allow_new_key:
    :return:
    """
    for k, v in dict2.items():
        if isinstance(v, dict) and '__overwrite__' in v.keys():
            overwrite = v.pop('__overwrite__')
        else:
            overwrite = False
        if isinstance(v, dict) and '__allow_new_key__' in v.keys():
            allow_new_key_child = v.pop('__allow_new_key__')
        else:
            allow_new_key_child = False
        if not allow_new_key and k not in dict1:
            logger.error(f'missing {k} from dict1, {dict1.keys()}')
            exit(1)
        if overwrite:
            logger.info(f'overwriting with key {k}')
            dict1[k] = v
        elif k not in dict1:
            logger.info(f'adding new key {k}')
            dict1[k] = v
        else:
            if isinstance(v, dict):
                assert isinstance(dict1[k], dict), (dict1[k], v)
                update_recursive(dict1[k], v, allow_new_key_child)
            else:
                assert not isinstance(dict1[k], dict), (dict1[k], v)
                dict1[k] = v
        #
        # if k in dict1:
        #     if isinstance(v, dict):
        #         if '__overwrite__' in v.keys():
        #             if v.pop('__overwrite__'):
        #                 dict1[k] = v
        #             else:
        #                 pass
        #         assert isinstance(dict1[k], dict), (dict1[k], v)
        #         update_recursive(dict1[k], v, allow_new_key)
        #     else:
        #         assert not isinstance(dict1[k], dict), (dict1[k], v)
        #         dict1[k] = v
        # else:
        #     if allow_new_key:
        #         dict1[k] = v
        #     else:
        #         logger.error(f'missing {k} from dict1, {dict1.keys()}')
        #         exit(1)

    return dict1, dict2
