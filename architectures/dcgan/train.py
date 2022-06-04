import torch
import torch.nn as nn
import torch.optim as optim
from . model import Generator, Discriminator
from .. common import BaseTrainer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DC_Trainer(BaseTrainer):
    def __init__(self, dataset, params: dict, num_workers=0):
        super().__init__(dataset, params, num_workers)
    
    def build(self, params):
        self.GEN = Generator(params)
        self.DISC = Discriminator(params)
        self.GEN.to(DEVICE)
        self.DISC.to(DEVICE)

        betas = (self.b1, self.b2)

        self.opt_gen = optim.Adam(self.GEN.parameters(), lr=self.lr_g, betas=betas)
        self.opt_disc = optim.Adam(self.DISC.parameters(), lr=self.lr_d, betas=betas)
        self.criterion = nn.BCELoss()
    
    def process_batch(self, value, labels):
        real = value.to(DEVICE)
        ## Get a batch of noise and run it through G to get a fake batch
        z = torch.randn(self.batch_size, self.latent_size, 1, 1, device=DEVICE)
        fake = self.GEN(z)

        ## Run the real batch through D and compute D's real loss
        D_real = self.DISC(real)
        loss_D_real = self.criterion(D_real, torch.ones_like(D_real))

        ## Run the fake batch through D and compute D's fake loss
        D_fake = self.DISC(fake.detach())
        loss_D_fake = self.criterion(D_fake, torch.zeros_like(D_fake))

        ## Get D's total loss
        loss_D = (loss_D_real + loss_D_fake)/2
        ## Train D
        self.DISC.zero_grad()
        loss_D.backward()
        self.opt_disc.step()

        ## Rerun the fake batch through trained D, then train G
        Dp_fake = self.DISC(fake)
        loss_G = self.criterion(Dp_fake, torch.ones_like(Dp_fake))
        
        self.GEN.zero_grad()
        loss_G.backward()
        self.opt_gen.step()

        return loss_G.item(), loss_D.item(), loss_D_real.item(), loss_D_fake.item()
