import math
import torch
import torch.nn as nn

#### model utilities ####


def check_required_params(o, params):
    for p in o.get_required_params():
        if p not in params:
            raise KeyError(p) ## should be caught by whatever tries to instanciate o

def grids_from_params_or_default(params: dict):
        """Ensure that params contain layer size transitions for G and D. 
        Generate meaningful defaults otherwise.

        Args:
            params (dict): model hyperparameters

        Raises:
            AttributeError: img_size needs to be known (or the params are malformed anyway)

        Returns:
            list, list : the layer size transitions
        """
        if "img_size" not in params:
            raise AttributeError
        img_size = params["img_size"]

        if "grids_g" not in params:
            grids_g = [1]
            i = 4
            while i <= img_size:
                grids_g.append(i)
                i *= 2
            params["grids_g"] = grids_g
        else:
            validate_grids(params["grids_g"])

        assert(params["grids_g"][-1] == img_size) # no weird stuff happened while building grids_g ?

        if "grids_d" not in params:
            grids_d = grids_g[::-1]
            params["grids_d"] = grids_d
        else:
            validate_grids(params["grids_d"])

        assert(params["grids_d"][0] == img_size) # ditto

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
