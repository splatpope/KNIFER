import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .model import Generator, Discriminator
from ..common import BaseTrainer

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def _init_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

class DC_Trainer(BaseTrainer):
    def __init__(self, dataset, params: dict, num_workers=0):
        super(DC_Trainer, self).__init__(dataset, params, num_workers)
    
    def build(self, params):
        self.GEN = Generator(params, features=self.features)
        _init_weights(self.GEN)
        self.DISC = Discriminator(params, features=self.features)
        _init_weights(self.DISC)
        self.GEN.to(DEVICE)
        self.DISC.to(DEVICE)

        betas = (self.b1, self.b2)

        self.opt_gen = optim.Adam(self.GEN.parameters(), lr=self.lr_g, betas=betas)
        self.opt_disc = optim.Adam(self.DISC.parameters(), lr=self.lr_d, betas=betas)
        self.criterion = nn.BCELoss()
    
    def process_batch(self, value, labels):
        x = value.to(DEVICE)
        ## Get a batch of noise and run it through G to get a fake batch
        z = torch.randn(self.batch_size, self.latent_size, 1, 1, device=DEVICE)
        g_z = self.GEN(z)
        ## Run the real batch through D and compute D's real loss
        d_x = self.DISC(x)
        loss_d_x = self.criterion(d_x, torch.ones_like(d_x))
        ## Run the fake batch through D and compute D's fake loss
        d_g_z = self.DISC(g_z.detach())
        loss_d_g_z = self.criterion(d_g_z, torch.zeros_like(d_g_z))
        ## Get D's total loss
        loss_d = (loss_d_x + loss_d_g_z)/2
        ## Train D
        self.DISC.zero_grad()
        loss_d.backward()
        self.opt_disc.step()
        ## Rerun the fake batch through trained D, then train G
        output = self.DISC(g_z)
        loss_g = self.criterion(output, torch.ones_like(output))
        self.GEN.zero_grad()
        loss_g.backward()
        self.opt_gen.step()
