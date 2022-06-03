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
        x = value.to(DEVICE)
        ## Get a batch of noise and run it through G to get a fake batch
        z = torch.randn(self.batch_size, self.latent_size, 1, 1, device=DEVICE)
        g_z = self.GEN(z)
        ## Run the real batch through D and compute D's real loss
        d_x = self.DISC(x)
        loss_d_x = F.relu(1.0 - d_x).mean()
        ## Run the fake batch through D and compute D's fake loss
        d_g_z = self.DISC(g_z.detach())
        loss_d_g_z = F.relu(1.0 + d_g_z).mean()
        ## Get D's total loss
        loss_d = loss_d_x + loss_d_g_z
        ## Train D
        self.DISC.zero_grad()
        loss_d.backward()
        self.opt_disc.step()

        ## Rerun a fake batch through trained D, then train G
        output = self.DISC(g_z)
        loss_g = -output.mean()
        self.GEN.zero_grad()
        loss_g.backward()
        self.opt_gen.step()

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
