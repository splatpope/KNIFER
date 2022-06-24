from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

from knifer.config import params as P

def NOOP(x):
    return x

class UpKConv(nn.ConvTranspose2d):
    def __init__(self, in_channels:int, out_channels:int, factor:int, **kwargs):
        self.factor = factor
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=2*factor,
            stride=factor,
            padding=factor//2,
            **kwargs,
        )
    def extra_repr(self):
        s = '{in_channels}, {out_channels}, factor={factor}'
        return s.format(**self.__dict__)

class DownKConv(nn.Conv2d):
    def __init__(self, in_channels:int, out_channels:int, factor:int, **kwargs):
        self.factor = factor
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=2*factor,
            stride=factor,
            padding=factor//2,
            **kwargs,
        )
    def extra_repr(self):
        s = '{in_channels}, {out_channels}, factor={factor}'
        return s.format(**self.__dict__)

class Attention(nn.Module):
    def __init__(self, channels:int):
        super().__init__()
        self.channels = channels
        self.theta = nn.Conv2d(channels, channels//8, 1, bias=False)
        self.phi = nn.Conv2d(channels, channels//8, 1, bias=False)
        self.g = nn.Conv2d(channels, channels//2, 1, bias=False)
        self.o = nn.Conv2d(channels//2, channels, 1, bias=False)
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor):
        N, C, W, H = x.shape

        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), 2)
        g = F.max_pool2d(self.g(x), 2)

        theta = theta.view(N, C//8, W*H)
        phi = phi.view(N, C//8, W*H//4)
        g = g.view(N, C//2, W*H//4)

        beta = F.softmax(torch.bmm(theta.transpose(1,2), phi), dim=-1)
        o_in = torch.bmm(g, beta.transpose(1,2)).view(N, C//2, W, H)
        o = self.o(o_in)
        return self.gamma * o + x
