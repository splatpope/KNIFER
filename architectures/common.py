import math
import torch
import torch.nn as nn

## A list of integer grid sizes is valid if they are all powers of 2 and are monotonically arranged
def validate_grids(grids):
    assert(isinstance(grids, list))
    assert(all([isinstance(g, int) for g in grids]))
    if any([not math.log2(g).is_integer() for g in grids]):
        return False
    if grids == sorted(grids) or grids == sorted(grids, reverse=True):
        return True
    return False

## Compute grid transitions to get layer parameters
def layers(grids):
    transitions = []
    if grids[0] < grids[1]: # upscaling
        for i in range(len(grids)-1):
            transitions.append(int(grids[i+1]/grids[i]))
    else: # downscaling
        for i in range(len(grids)-1):
            transitions.append(int(grids[i]/grids[i+1]))
    return transitions

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
