from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

from knifer.config import params as P

def NOOP(x):
    return x

class ConvBlock(nn.Module):
    def __init__(self, params: P.ConvBlockParameters):
        super().__init__()
        self.conv: Union[nn.Conv2d, nn.ConvTranspose2d] = params.conv_module(
            in_channels=params.in_channels,
            out_channels=params.out_channels,
            kernel_size=params.kernel_size,
            stride=params.stride,
            padding=params.padding,
            bias=not isinstance(params.norm_layer, nn.BatchNorm2d),
        )
        if params.spectral_norm:
            self.conv = spectral_norm(self.conv)

        self.norm = params.norm_layer(params.out_channels) if params.norm_layer else NOOP
        self.act_fn = params.activation if params.activation else NOOP
        self.attn = Attention(params.out_channels) if params.self_attention else NOOP
        
    def forward(self, input):
        out = self.conv(input)
        out = self.norm(out)
        out = self.act_fn(out)
        out = self.attn(out)
        return out
    
class UpKConv(nn.ConvTranspose2d):
    def __init__(self, factor:int, *args, **kwargs):
        super().__init__(
            kernel_size=2*factor,
            stride=factor,
            padding=factor//2,
            *args, **kwargs,
        )

class DownKConv(nn.Conv2d):
    def __init__(self, factor:int, *args, **kwargs):
        super().__init__(
            kernel_size=2*factor,
            stride=factor,
            padding=factor//2,
            *args, **kwargs,
        )

class Attention(nn.Module):
    def __init__(self, channels):
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
