from torchvision.utils import make_grid
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from tu.ddp import all_gather


def get_image(t, **kwargs):
    # expect t in range (0, 1)
    grid = make_grid(t, **kwargs)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return Image.fromarray(ndarr)


def dump_helper(self, k, v, prefix=None):
    # helper function
    if dist.is_initialized():
        v = all_gather(v.contiguous())
        if dist.get_rank() != 0:
            return
    if prefix is not None:
        k = f'{prefix}/{k}'
    self.vi_helper.dump_table(self.vi, [[get_image(v)]], table_name='', col_names=[k])
    self.writer.add_image(k, make_grid(v).clamp(0, 1), self.it)


def dump_row_helper(self, ks, vs):
    assert len(ks) == len(vs), (len(ks), len(vs), ks)

    if dist.is_initialized():
        vs = [all_gather(v.contiguous()) for v in vs]
        if dist.get_rank() != 0:
            return
    self.vi_helper.dump_table(self.vi, [[get_image(v) for v in vs]], table_name='', col_names=ks)
    for k, v in zip(ks, vs):
        self.writer.add_image(k, make_grid(v).clamp(0, 1), self.it)


def to_frames(t, fps=32, **kwargs):
    w, h = t[0].size
    if w % 2 != 0 or h % 2 != 0:
        t = map(lambda tt: tt.resize((w // 2 * 2, h // 2 * 2)), t)
    return dict(video=list(map(np.array, t)), fps=fps, **kwargs)
