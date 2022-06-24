from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from knifer.misc_utils import gradient_penalty

# l, li
# ll, l_

class BaseGANLoss():
    def __init__(self, device:str='cpu'):
        self.device = device
    def D(self, D_real: torch.Tensor, D_fake: torch.Tensor) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
        raise NotImplementedError
    def G(self, D_fake: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class BCE(BaseGANLoss):
    lossfn = F.binary_cross_entropy
    def __init__(self, device:str="cpu"):
        super().__init__(device)

    def D(self, D_real, D_fake):
        l_d_real = BCE.lossfn(D_real, torch.ones_like(D_real, device=self.device))
        l_d_fake = BCE.lossfn(D_fake, torch.zeros_like(D_fake, device=self.device))
        return (l_d_real + l_d_fake)/2, l_d_real.item(), l_d_fake.item()

    def G(self, D_fake):
        return BCE.lossfn(D_fake, torch.ones_like(D_fake))

# TODO : this should just be means and GP should be a regularization technique
class WGP(BaseGANLoss):
    def __init__(self, critic:nn.Module, device:str="cpu"):
        super().__init__(device)
        self.critic = critic
    
    def D(self, D_real, D_fake):
        l_d_real = -torch.mean(D_real)
        l_d_fake = torch.mean(D_fake)
        gp = gradient_penalty(self.critic, D_real, D_fake, self.device)
        return l_d_real + l_d_fake, l_d_real.item(), l_d_fake.item()
    
    def G(self, D_fake):
        return -torch.mean(D_fake)

class Hinge(BaseGANLoss):
    def __init__(self, device:str="cpu"):
        super().__init__(device)

    def D(self, D_real, D_fake):
        l_d_real = F.relu(1.0 - D_real).mean()
        l_d_fake = F.relu(1.0 + D_fake).mean()
        return l_d_real + l_d_fake, l_d_real.item(), l_d_fake.item()
    
    def G(self, D_fake):
        return -torch.mean(D_fake)

STR_TO_LOSS = {
    "dcgan": BCE,
    "bce": BCE,
    "wasserstein": WGP,
    "w": WGP,
    "hinge": Hinge,
}