import torch
import torch.nn as nn
import torch.nn.utils.parametrizations as nnp

def spectral_norm(target: nn.Module):
    is_conv = lambda m: isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))
    if isinstance(target, (nn.Sequential, nn.ModuleList)):
        modules = {i: nnp.spectral_norm(m) for i, m in enumerate(target)}
        for i, m in modules.items():
            target[i] = m

# TODO : init methods
