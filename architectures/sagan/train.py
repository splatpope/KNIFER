import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from architectures.common import check_required_params
from .model import Generator, Discriminator
from ..dcgan.train import Trainer as DCGANTrainer
from ..wgan_gp.train import Trainer as WGAN_GPTrainer
from ..wgan_gp.util import gradient_penalty

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _init_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

class Trainer(DCGANTrainer):
    def __init__(self, dataset, params: dict):
        check_required_params(self, params)
        super(Trainer, self).__init__(dataset, params)

    def build(self, params):
        self.GEN = Generator(params)
        _init_weights(self.GEN)
        self.DISC = Discriminator(params)
        _init_weights(self.DISC)
        self.GEN.to(DEVICE)
        self.DISC.to(DEVICE)

        betas = (self.b1, self.b2)

        self.opt_gen = optim.Adam(self.GEN.parameters(), lr=self.learning_rate, betas=betas)
        self.opt_disc = optim.Adam(self.DISC.parameters(), lr=self.learning_rate, betas=betas)
        self.criterion = nn.BCELoss()

    @classmethod
    def get_required_params(cls):
        return super().get_required_params() + [
            "attn_spots"
        ]

class WGPTrainer(WGAN_GPTrainer):
    def __init__(self, dataset, params: dict):
        check_required_params(self, params)
        super(WGPTrainer, self).__init__(dataset, params)

    def build(self, params):
        self.GEN = Generator(params)
        self.DISC = Discriminator(params)
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
