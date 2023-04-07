import importlib
from typing import Dict, Any
import fnmatch
from tu.configs import nested_dict_to_dot_map_dict
import logging


logger = logging.getLogger(__name__)


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def build_from_config(config, **kwargs):
    if config is None:
        return None
    if isinstance(config, str):
        return get_obj_from_str(config)
    obj = get_obj_from_str(config['__target__'])
    return obj(**config['kwargs'], **kwargs)


def build_from_config_recursive(config, **kwargs):
    if config is None:
        return None
    if not isinstance(config, dict) or '__target__' not in config:
        return config
    obj = get_obj_from_str(config['__target__'])
    kwargs_full = dict()
    for k, v in config['kwargs'].items():
        if isinstance(v, (tuple, list)):
            v = type(v)(map(build_from_config_recursive, v))
        elif isinstance(v, dict):
            if isinstance(v, dict) and '__target__' in v and 'kwargs' in v:
                v = build_from_config_recursive(v)
            else:
                v = {k2: build_from_config_recursive(v2) for k2, v2 in v.items()}
        kwargs_full[k] = v
    for k, v in kwargs.items():
        assert k not in kwargs_full, (k, v, kwargs_full)
        kwargs_full[k] = v
    return obj(**kwargs_full)


def check_cfg_consistency(cfg1, cfg2, ignore_keys=None):
    if ignore_keys is None:
        ignore_keys = set()
    cfg1 = nested_dict_to_dot_map_dict(cfg1)
    cfg2 = nested_dict_to_dot_map_dict(cfg2)

    def get_keys(cfg):
        keys = set()
        for k in cfg.keys():
            k_found = False
            for k_ignore in ignore_keys:
                if fnmatch.fnmatch(k, k_ignore):
                    k_found = True
                    break
            if not k_found:
                keys.add(k)
        return keys

    keys1, keys2 = get_keys(cfg1), get_keys(cfg2)

    logger.info(f'keys diff: {keys1 - keys2, keys2 - keys2, ignore_keys}')

    ret = True

    for k in sorted(keys1.union(keys2)):
        if k not in cfg1:
            ret = False
            logger.error(f'inconsistent key: {k}, missing, {cfg2[k]}')
        elif k not in cfg2:
            ret = False
            logger.error(f'inconsistent key: {k}, {cfg1[k]}, missing')
        elif cfg1[k] != cfg2[k]:
            ret = False
            logger.error(f'inconsistent key: {k}, {cfg1[k]}, {cfg2[k]}')

    return ret


def overwrite_cfg(cfg: Dict, key: str, value: Any, recursive=False, check_exists=True):
    if check_exists:
        assert key in cfg, key
    if key in cfg and recursive and isinstance(value, dict):
        for k, v in value.items():
            overwrite_cfg(cfg[key], k, v, recursive=recursive, check_exists=check_exists)
    else:
        logger.info(f'overwrite key {key}: {cfg.get(key)} -> {value}')
        cfg[key] = value
    return cfg


def overwrite_cfg_if_not_exist(cfg: Dict, key: str, value: Any):
    if key not in cfg:
        logger.info(f'key {key} not found')
        overwrite_cfg(cfg, key, value, check_exists=False)
    return cfg
