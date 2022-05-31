import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from architectures.common import BaseTrainer
from .model import Generator, Discriminator
from .util import gradient_penalty

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def _init_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

class WGP_Trainer(BaseTrainer):
    def __init__(self, dataset, params, num_workers):
        self.critic_iters = params["critic_iters"]
        self.lambda_gp = params["lambda_gp"]
        super(WGP_Trainer, self).__init__(dataset, params, num_workers)

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
        #self.criterion = nn.BCELoss()

    def process_batch(self, value, labels):
        x = value.to(DEVICE) # real
        for i in range(self.critic_iters):
            ## Get a batch of noise and run it through G to get a fake batch
            z = torch.randn(value.shape[0], self.latent_size, 1, 1, device=DEVICE)
            g_z = self.GEN(z) ## fake

            ## Run the real batch through D and compute D's real loss
            d_x = self.DISC(x) ## critic real

            #loss_d_x = self.criterion(d_x, torch.ones_like(d_x)) ## not needed for wgan
            ## Run the fake batch through D and compute D's fake loss
            d_g_z = self.DISC(g_z.detach()) ## critic fake
            #loss_d_g_z = self.criterion(d_g_z, torch.zeros_like(d_g_z)) ## not needed for wgan
            gp = gradient_penalty(self.DISC, x, g_z, device=DEVICE)
            loss_critic = (
                -(torch.mean(d_x) - torch.mean(d_g_z)) + self.lambda_gp * gp
            )

            self.DISC.zero_grad()
            loss_critic.backward(retain_graph=True)
            self.opt_disc.step()

        ## Rerun the fake batch through trained D, then train G
        output = self.DISC(g_z) # critic fake post training
        loss_g = -torch.mean(output)
        self.GEN.zero_grad()
        loss_g.backward()
        self.opt_gen.step()

    @classmethod
    def get_required_params(cls):
        return super().get_required_params() + [
            "critic_iters",
            "lambda_gp",
        ]
