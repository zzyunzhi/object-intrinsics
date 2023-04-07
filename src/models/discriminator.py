import torch.nn as nn
import torch
from tu.utils.config import build_from_config
from math import log2
import torch.nn.functional as F


class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout,
                                3, stride=1, padding=1, bias=is_bias)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(
                self.fin, self.fout, 1, stride=1, padding=0, bias=False)

    def actvn(self, x):
        out = F.leaky_relu(x, 2e-1)
        return out

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(self.actvn(x))
        dx = self.conv_1(self.actvn(dx))
        out = x_s + 0.1*dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


class DCDiscriminator(nn.Module):
    ''' DC Discriminator class.

    Args:
        in_dim (int): input dimension
        n_feat (int): features of final hidden layer
        img_size (int): input image size
    '''
    def __init__(self, in_dim=3, out_dim=1, n_feat=512, img_size=64, last_bias=False):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        n_layers = int(log2(img_size) - 2)
        self.blocks = nn.ModuleList(
            [nn.Conv2d(
                in_dim,
                int(n_feat / (2 ** (n_layers - 1))),
                4, 2, 1, bias=False)] + [nn.Conv2d(
                    int(n_feat / (2 ** (n_layers - i))),
                    int(n_feat / (2 ** (n_layers - 1 - i))),
                    4, 2, 1, bias=False) for i in range(1, n_layers)])

        self.conv_out = nn.Conv2d(n_feat, out_dim, 4, 1, 0, bias=last_bias)
        self.actvn = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, **kwargs):
        batch_size = x.shape[0]
        if x.shape[1] != self.in_dim:
            import ipdb; ipdb.set_trace()
            x = x[:, :self.in_dim]
        for layer in self.blocks:
            x = self.actvn(layer(x))

        out = self.conv_out(x)
        out = out.reshape(batch_size, self.out_dim)
        return out


class ADADiscriminator(DCDiscriminator):
    def __init__(self, aug, aug_p, **kwargs):
        super().__init__(**kwargs)
        self.aug = build_from_config(aug)
        self.aug.p.copy_(torch.tensor(aug_p, dtype=torch.float32))
        self.resolution = kwargs['img_size']

    def get_resolution(self):
        return self.resolution

    def forward(self, x, **kwargs):
        x = self.aug(x)
        return super().forward(x, **kwargs)


class ADADiscriminatorView(ADADiscriminator):
    def __init__(self, out_dim_position, out_dim_latent, **kwargs):
        self.out_dim_position = out_dim_position
        self.out_dim_latent = out_dim_latent

        super().__init__(**kwargs)
