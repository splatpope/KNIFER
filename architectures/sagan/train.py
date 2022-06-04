import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from architectures.common import BaseTrainer
from .model import Generator, Discriminator
from ..wgan_gp.train import WGP_Trainer as WGAN_GPTrainer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class SA_Trainer(BaseTrainer):
    def __init__(self, dataset, params: dict, num_workers):
        super(SA_Trainer, self).__init__(dataset, params, num_workers)

    def build(self, params):
        self.GEN = Generator(params)
        self.DISC = Discriminator(params)
        self.GEN.to(DEVICE)
        self.DISC.to(DEVICE)

        betas = (self.b1, self.b2)

        self.opt_gen = optim.Adam(filter(lambda p: p.requires_grad, self.GEN.parameters()), lr=self.lr_g, betas=betas)
        self.opt_disc = optim.Adam(filter(lambda p: p.requires_grad, self.DISC.parameters()), lr=self.lr_d, betas=betas)
        #self.criterion = nn.BCELoss()

#uses hinge loss
    def process_batch(self, value, labels):
        real = value.to(DEVICE)
        ## Get a batch of noise and run it through G to get a fake batch
        z = torch.randn(self.batch_size, self.latent_size, 1, 1, device=DEVICE)
        fake = self.GEN(z)

        ## Run the real batch through D and compute D's real loss
        D_real = self.DISC(real)
        loss_D_real = F.relu(1.0 - D_real).mean()

        ## Run the fake batch through D and compute D's fake loss
        D_fake = self.DISC(fake.detach())
        loss_D_fake = F.relu(1.0 + D_fake).mean()

        ## Get D's total loss
        loss_D = loss_D_real + loss_D_fake
        ## Train D
        self.DISC.zero_grad()
        loss_D.backward()
        self.opt_disc.step()

        ## Rerun a fake batch through trained D, then train G
        Dp_fake = self.DISC(fake)
        loss_G = -Dp_fake.mean()
        
        self.GEN.zero_grad()
        loss_G.backward()
        self.opt_gen.step()

        return loss_G.item(), loss_D.item(), loss_D_real.item(), loss_D_fake.item()

    @classmethod
    def get_required_params(cls):
        return super().get_required_params() + [
            "attn_spots"
        ]

class SA_WGP_Trainer(WGAN_GPTrainer):
    def __init__(self, dataset, params: dict, num_workers):
        super(SA_WGP_Trainer, self).__init__(dataset, params, num_workers)

    def build(self, params):
        self.GEN = Generator(params)
        self.DISC = Discriminator(params)
        self.GEN.to(DEVICE)
        self.DISC.to(DEVICE)

        betas = (self.b1, self.b2)

        self.opt_gen = optim.Adam(self.GEN.parameters(), lr=self.lr_g, betas=betas)
        self.opt_disc = optim.Adam(self.DISC.parameters(), lr=self.lr_d, betas=betas)
        #self.criterion = nn.BCELoss()

# TODO : not reuse the WGP train method, since we have TTUR now OR simply don't use critic_iters with sagan

    @classmethod
    def get_required_params(cls):
        return super().get_required_params() + [
            "attn_spots"
        ]
