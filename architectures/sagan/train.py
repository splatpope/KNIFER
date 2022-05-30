import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from architectures.common import check_required_params
from .model import Generator, Discriminator
from ..dcgan.train import Trainer as DCGANTrainer
from ..wgan_gp.train import Trainer as WGAN_GPTrainer
from ..wgan_gp.util import gradient_penalty

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def _init_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

class Trainer(DCGANTrainer):
    def __init__(self, dataset, params: dict, num_workers):
        check_required_params(self, params)
        super(Trainer, self).__init__(dataset, params, num_workers)

    def build(self, params):
        self.GEN = Generator(params, features=self.features)
        _init_weights(self.GEN)
        self.DISC = Discriminator(params, features=self.features)
        _init_weights(self.DISC)
        self.GEN.to(DEVICE)
        self.DISC.to(DEVICE)

        betas = (self.b1, self.b2)

        self.opt_gen = optim.Adam(self.GEN.parameters(), lr=self.learning_rate, betas=betas)
        self.opt_disc = optim.Adam(self.DISC.parameters(), lr=self.learning_rate, betas=betas)
        self.criterion = nn.BCELoss()

#uses hinge loss
    def process_batch(self, value, labels):
        x = value.to(DEVICE)
        ## Get a batch of noise and run it through G to get a fake batch
        z = torch.randn(self.batch_size, self.latent_size, 1, 1, device=DEVICE)
        g_z = self.GEN(z)
        ## Run the real batch through D and compute D's real loss
        d_x = self.DISC(x)
        loss_d_x = torch.nn.ReLU(1.0 - d_x).mean()
        ## Run the fake batch through D and compute D's fake loss
        d_g_z = self.DISC(g_z.detach())
        loss_d_g_z = torch.nn.ReLU(1.0 + d_g_z).mean()
        ## Get D's total loss
        loss_d = loss_d_x + loss_d_g_z
        ## Train D
        self.DISC.zero_grad(set_to_none=True)
        loss_d.backward()
        self.opt_disc.step()
        ## Rerun the fake batch through trained D, then train G
        output = self.DISC(g_z)
        loss_g = -output.mean()
        self.GEN.zero_grad(set_to_none=True)
        loss_g.backward()
        self.opt_gen.step()

    @classmethod
    def get_required_params(cls):
        return super().get_required_params() + [
            "attn_spots"
        ]

class WGPTrainer(WGAN_GPTrainer):
    def __init__(self, dataset, params: dict, num_workers):
        check_required_params(self, params)
        super(WGPTrainer, self).__init__(dataset, params, num_workers)

    def build(self, params):
        self.GEN = Generator(params, features=self.features)
        _init_weights(self.GEN)
        self.DISC = Discriminator(params, features=self.features)
        _init_weights(self.DISC)
        self.GEN.to(DEVICE)
        self.DISC.to(DEVICE)

        betas = (self.b1, self.b2)

        self.opt_gen = optim.Adam(self.GEN.parameters(), lr=self.learning_rate, betas=betas)
        self.opt_disc = optim.Adam(self.DISC.parameters(), lr=self.learning_rate, betas=betas)
        #self.criterion = nn.BCELoss()

    @classmethod
    def get_required_params(cls):
        return super().get_required_params() + [
            "attn_spots"
        ]
