import math
import torch
import torch.nn as nn

def ReSample(in_c, out_c, factor, bias=True, dir="up"):
    assert(factor >= 2)
    assert(math.log2(factor).is_integer())
    k = int(2*factor)
    s = int(factor)
    p = int(factor/2)
    if dir == "up":
        return nn.ConvTranspose2d(
            in_channels=in_c, 
            out_channels=out_c, 
            kernel_size=k, 
            stride=s, 
            padding=p, 
            bias=bias
        )
    elif dir == "down":
        return nn.Conv2d(
            in_channels=in_c, 
            out_channels=out_c, 
            kernel_size=k, 
            stride=s, 
            padding=p, 
            bias=bias
        )

def UpSample(in_c, out_c, factor, bias=True):
    return ReSample(in_c, out_c, factor, bias, dir="up")

def DownSample(in_c, out_c, factor, bias=True):
    return ReSample(in_c, out_c, factor, bias, dir="down")