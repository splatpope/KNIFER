import torch
import torch.nn as nn
import torch.nn.utils.parametrizations as nnp

def spectral_norm(target: nn.Module):
    """ Applies the spectral normalization parametrization to the module
        and all its submodules if any of them are 2D convolutions or module
        containers.
    """
    is_conv = lambda m: isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))
    if isinstance(target, (nn.Sequential, nn.ModuleList)):
        modules = {i: nnp.spectral_norm(m) for i, m in enumerate(target) if is_conv(m)}
        for i, m in modules.items():
            target[i] = m
    elif is_conv(target):
        target = nnp.spectral_norm(target)

# TODO : init methods
