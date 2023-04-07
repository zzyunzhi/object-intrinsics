import torch
import torch.nn as nn
import numpy as np
import random
import os
import subprocess
import atexit
import signal
import logging

logger = logging.getLogger(__name__)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def set_seed_benchmark(seed):
    # https://github.com/facebookresearch/deit/blob/ee8893c8063f6937fec7096e47ba324c206e22b9/main.py#L197
    logger.info(f'setting seed {seed}')
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    torch.cuda.manual_seed_all(seed)


def spawn_ddp(args, worker):
    """

    Args:
        worker: a function with argument rank, world_size, args_in
            example see test_ddp_spawn

    Returns:

    """
    assert torch.cuda.is_available()
    world_size = torch.cuda.device_count()

    torch.multiprocessing.spawn(worker, nprocs=world_size, args=(world_size, args), join=True)


def count_parameters(model: nn.Module, name='', verbose=True):
    params = model.parameters()
    param_count = sum(p.numel() for p in params)
    if verbose:
        print('param count', param_count, 'model name', name)
    return param_count


def count_trainable_parameters(model: nn.Module, name='', verbose=True):
    params = model.parameters()
    params = filter(lambda p: p.requires_grad, params)
    param_count = sum(p.numel() for p in params)
    if verbose:
        print('trainable param count', param_count, 'model name', name)
    return param_count


def count_not_trainable_parameters(model: nn.Module, name='', verbose=True):
    params = model.parameters()
    params = filter(lambda p: not p.requires_grad, params)
    param_count = sum(p.numel() for p in params)
    if verbose:
        print('not trainable param count', param_count, 'model name', name)
    return param_count


def open_tensorboard(log_dir):
    p = subprocess.Popen(
        ["tensorboard", "--logdir", log_dir, '--bind_all', '--reload_multifile', 'True', '--load_fast', 'false']
    )

    def killme():
        os.kill(p.pid, signal.SIGTERM)

    atexit.register(killme)
