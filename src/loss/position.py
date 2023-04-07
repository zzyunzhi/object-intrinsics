import torch.nn.functional as F


class PositionLoss:
    def __init__(self, loss_str):
        if loss_str == 'mse':
            self.loss = F.mse_loss
        elif loss_str == 'smooth_l1':
            self.loss = F.smooth_l1_loss

    def __call__(self, pred, target, reduction='mean'):
        return self.loss(pred, target, reduction=reduction)


def linear_increase(max_it, max_weight):
    def fn(it):
        return min(it / max_it, 1) * max_weight
    return fn
