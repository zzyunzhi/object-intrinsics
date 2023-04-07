from torch import autograd
import torch.nn.functional as F


def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.reshape(batch_size, -1).sum(1)
    return reg.mean()


# WARNINGS: do not use sigmoid activation for the last layer of the discriminator

def _compute_bce(d_out, target):
    targets = d_out.new_full(size=d_out.size(), fill_value=target)
    loss = F.binary_cross_entropy_with_logits(d_out, targets)
    return loss


def _compute_mse(d_out, target):
    targets = d_out.new_full(size=d_out.size(), fill_value=target)
    loss = F.mse_loss(d_out, targets)
    return loss


def _compute_wgangp(d_out, target):
    if target == 1:  # real
        return -d_out.mean()
    elif target == 0:  # fake
        return d_out.mean()
    raise RuntimeError(f'invalid target value: {target}')


class GANLoss:
    def __init__(self, gan_str: str):
        self.fn = {
            'bce': _compute_bce,
            'mse': _compute_mse,
            'wgangp': _compute_wgangp,
        }[gan_str]

    def __call__(self, d_out, target):
        assert len(d_out.shape) == 2 and d_out.shape[1] == 1, d_out.shape
        return self.fn(d_out, target)

