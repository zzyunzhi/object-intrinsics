import torch.nn as nn
from typing import Optional
from torch.nn.parallel import DistributedDataParallel
import copy


class EMA:
    def __init__(self, m: nn.Module, beta: float, m_ema: Optional[nn.Module] = None):
        if isinstance(m, DistributedDataParallel):
            m = m.module
        if m_ema is None:
            m_ema = copy.deepcopy(m)

        m_ema = m_ema.eval()
        for p in m_ema.parameters():
            p.requires_grad = False

        self.m = m
        self.m_ema = m_ema
        self.beta = beta

    @property
    def module(self):
        return self.m_ema

    def update(self, it):
        beta = self.beta # min(self.beta, 0.5 ** (200 / max(it, 1000)))

        for p_ema, p in zip(self.m_ema.parameters(), self.m.parameters()):
            p_ema.copy_(p.lerp(p_ema, beta))
        for p_ema, p in zip(self.m_ema.buffers(), self.m.buffers()):
            p_ema.copy_(p)

    def get_state_dict(self):
        return {
            'state_dict': self.m_ema.state_dict(),
            'beta': self.beta,
        }

    def __str__(self):
        return f'ema@{self.beta}'
