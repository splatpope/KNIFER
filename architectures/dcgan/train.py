import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .model import Generator, Discriminator
from ..common import check_required_params, grids_from_params_or_default

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer():
    def __init__(self, dataset, params: dict):
        check_required_params(self, params)
        self.batch_size = params["batch_size"]
        self.latent_size = params["latent_size"]
        self.learning_rate = params["learning_rate"]
        self.b1 = params["b1"]
        self.b2 = params["b2"]

        sample = dataset[0][0]

        self.img_size = sample.shape[2]
        if not "img_size" in params: ## should never happen but hey
            params["img_size"] = self.img_size

        self.channels = sample.shape[0]
        if not "img_channels" in params:
            params["img_channels"] = self.channels

        grids_from_params_or_default(params)

        self.data = DataLoader(dataset, self.batch_size, shuffle=True)

        #self.state = "ready"
    
    def build(self, params):
        self.GEN = Generator(params)
        self.DISC = Discriminator(params)
        self.GEN.to(DEVICE)
        self.DISC.to(DEVICE)

        betas = (self.b1, self.b2)

        self.opt_gen = optim.Adam(self.GEN.parameters(), lr=self.learning_rate, betas=betas)
        self.opt_disc = optim.Adam(self.DISC.parameters(), lr=self.learning_rate, betas=betas)
        self.criterion = nn.BCELoss()

    @classmethod
    def get_required_params(cls):
        return [
            "batch_size",
            "latent_size",
            "learning_rate",
            "b1",
            "b2",
        ]
    
    def process_batch(self, value, labels):
        x = value.to(DEVICE)
        ## Get a batch of noise and run it through G to get a fake batch
        z = torch.randn(self.batch_size, self.latent_size, 1, 1).to(DEVICE)
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

    def get_fixed(self):
        return torch.randn(self.batch_size, self.latent_size, 1, 1).to(DEVICE)

    def serialize(self):
        return {
            'G_state': self.GEN.state_dict(),
            'D_state': self.DISC.state_dict(),
            'optG_state': self.opt_gen.state_dict(),
            'optD_state':self.opt_disc.state_dict(),
        }

    def deserialize(self, state):
        self.GEN.load_state_dict(state['G_state'])
        self.DISC.load_state_dict(state['D_state'])
        self.opt_gen.load_state_dict(state['optG_state'])
        self.opt_disc.load_state_dict(state['optD_state'])
