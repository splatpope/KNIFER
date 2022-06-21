from typing import Union
import torch
import torch.nn as nn
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
        
    def forward(self, input):
        out = self.conv(input)
        out = self.norm(out)
        out = self.act_fn(out)
        return out
    

