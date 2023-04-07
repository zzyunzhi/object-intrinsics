import torch
import torch.nn as nn
import re
import logging
import torch.distributed as dist
import collections
import torch


logger = logging.getLogger(__name__)


def get_grad_norm(m: nn.Module):
    grad_norm = [torch.linalg.norm(p.grad) for p in m.parameters() if p.requires_grad]
    if len(grad_norm) == 0:
        return None
    return torch.stack(grad_norm).mean()


def get_children_grad_norm(m: nn.Module):
    return {k: get_grad_norm(v) for k, v in m.named_children()}


def get_children_grad_norm_safe(m: nn.Module, verbose=True):
    ret = dict()
    for k, v in m.named_children():
        grad_norm = []
        for n, p in v.named_parameters():
            if not p.requires_grad:
                continue
            if p.grad is None:
                if verbose:
                    logger.error(f'{k}.{n} has no grad')
                continue
            grad_norm.append(torch.linalg.norm(p.grad))
        if len(grad_norm) == 0:
            grad_norm = None
        else:
            grad_norm = torch.stack(grad_norm).mean()
        ret[k] = grad_norm
    return ret


def get_optimizer_lr(optimizer: torch.optim.Optimizer):
    ret = dict()
    for param_ind, param_group in enumerate(optimizer.param_groups):
        ret[param_group.get('name', param_ind)] = param_group['lr']
    return ret


def recursive_map_data(fn, data):
    r"""Move all tensors inside data to device.
    Args:
        fn: mapping
        data (dict, list, or tensor): Input data.
    """
    if isinstance(data, torch.Tensor):
        return fn(data)
    elif isinstance(data, collections.abc.Mapping):
        return {k: recursive_map_data(fn, v) for k, v in data.items()}
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, (str, bytes)):
        return [recursive_map_data(fn, v) for v in data]
    else:
        return data


def process_batch(batch):
    if dist.is_initialized():
        rank = dist.get_rank()
        fn = lambda v: v.cuda(rank, non_blocking=True)
    else:
        fn = lambda v: v.cuda()
    return recursive_map_data(fn, batch)
