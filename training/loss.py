import torch
import torch.nn as nn
import torch.nn.functional as F

from misc_utils import gradient_penalty

# l, li
# ll, l_

class BaseLoss():
    def __init__(self, device='cpu'):
        self.device = device
    def D(self):
        raise NotImplementedError
    def G(self):
        raise NotImplementedError

class BCELoss(BaseLoss):
    def __init__(self, device="cpu"):
        super().__init__(device)

    def D(self, D_real, D_fake):
        bce = F.binary_cross_entropy
        l_d_real = bce(D_real, torch.ones_like(D_real, device=self.device))
        l_d_fake = bce(D_fake, torch.zeros_like(D_fake, device=self.device))
        return (l_d_real + l_d_fake)/2, l_d_real.item(), l_d_fake.item()

    def G(self, D_fake):
        return F.binary_cross_entropy(D_fake, torch.ones_like(D_fake))

class WGPLoss(BaseLoss):
    def __init__(self, critic: nn.Module, device="cpu"):
        super().__init__(device)
        self.critic = critic
    
    def D(self, D_real, D_fake):
        l_d_real = -torch.mean(D_real)
        l_d_fake = torch.mean(D_fake)
        gp = gradient_penalty(self.critic, D_real, D_fake, self.device)
        return l_d_real + l_d_fake, l_d_real.item(), l_d_fake.item()
    
    def G(self, D_fake):
        return -torch.mean(D_fake)

class HingeLoss(BaseLoss):
    def __init__(self, device="cpu"):
        super().__init__(device)

    def D(self, D_real, D_fake):
        l_d_real = F.relu(1.0 - D_real).mean()
        l_d_fake = F.relu(1.0 + D_fake).mean()
        return l_d_real + l_d_fake, l_d_real.item(), l_d_fake.item()
    
    def G(self, D_fake):
        return -torch.mean(D_fake)
