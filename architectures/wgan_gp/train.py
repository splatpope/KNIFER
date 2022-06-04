import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from architectures.common import BaseTrainer
from .model import Generator, Discriminator
from .util import gradient_penalty

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class WGP_Trainer(BaseTrainer):
    def __init__(self, dataset, params, num_workers):
        self.critic_iters = params["critic_iters"]
        self.lambda_gp = params["lambda_gp"]
        super(WGP_Trainer, self).__init__(dataset, params, num_workers)

    def build(self, params):
        self.GEN = Generator(params, features=self.features)
        self.DISC = Discriminator(params, features=self.features)

        self.GEN.to(DEVICE)
        self.DISC.to(DEVICE)

        betas = (self.b1, self.b2)

        self.opt_gen = optim.Adam(self.GEN.parameters(), lr=self.lr_g, betas=betas)
        self.opt_disc = optim.Adam(self.DISC.parameters(), lr=self.lr_d, betas=betas)
        #self.criterion = nn.BCELoss()

    def process_batch(self, value, labels):
        real = value.to(DEVICE)
        for i in range(self.critic_iters):
            ## Get a batch of noise and run it through G to get a fake batch
            z = torch.randn(value.shape[0], self.latent_size, 1, 1, device=DEVICE)
            fake = self.GEN(z) ## fake

            ## Run the real batch through D and compute D's real loss
            D_real = self.DISC(real) ## critic real
            loss_D_real = -torch.mean(D_real)
            ## Run the fake batch through D and compute D's fake loss
            D_fake = self.DISC(fake.detach()) ## critic fake
            loss_D_fake = torch.mean(D_fake)
            gp = gradient_penalty(self.DISC, real, fake, device=DEVICE)
            loss_critic = (
                loss_D_real + loss_D_fake + self.lambda_gp * gp
            )

            self.DISC.zero_grad()
            loss_critic.backward(retain_graph=True)
            self.opt_disc.step()

        ## Rerun the fake batch through trained D, then train G
        Dp_fake = self.DISC(fake) # critic fake post training
        loss_G = -torch.mean(Dp_fake)
        self.GEN.zero_grad()
        loss_G.backward()
        self.opt_gen.step()

        return loss_G.item(), loss_critic.item(), loss_D_real.item(), loss_D_fake.item()

    @classmethod
    def get_required_params(cls):
        return super().get_required_params() + [
            "critic_iters",
            "lambda_gp",
        ]
