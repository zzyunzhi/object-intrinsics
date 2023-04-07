from contextlib import contextmanager
import time
import logging

logger = logging.getLogger(__name__)


@contextmanager
def print_time(name) -> float:
    t = time.perf_counter()
    yield
    logger.info(f'{name}: {time.perf_counter() - t:.4f}s')


@contextmanager
def named_timeit(name, store_dict) -> float:
    if name not in store_dict:
        store_dict[name] = 0
    t = time.perf_counter()
    yield
    store_dict[name] += time.perf_counter() - t
    
    
@contextmanager
def timeit_as_list(name, store_dict) -> float:
    if name not in store_dict:
        store_dict[name] = []
    t = time.perf_counter()
    yield
    store_dict[name].append(time.perf_counter() - t)
