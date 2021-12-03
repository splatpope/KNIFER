import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .model import Generator, Discriminator
from .util import gradient_penalty

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer():
    def __init__(self, dataset, batch_size=16, latent_size=100, learning_rate=2e-4, b1=0.0, b2=0.9, critic_iters=5, lambda_gp = 10):
        self.batch_size = batch_size
        self.lr = learning_rate
        self.z_size = latent_size
        self.critic_iters = critic_iters
        self.lambda_gp = lambda_gp

        sample = dataset[0][0]
        self.img_size = sample.shape[2]
        self.channels = sample.shape[0]
        print(self.channels)

        self.data = DataLoader(dataset, self.batch_size, shuffle=True)

        self.GEN = Generator(self.img_size, self.channels, self.z_size)
        self.CRITIC = Discriminator(self.img_size, self.channels)
        self.GEN.to(DEVICE)
        self.CRITIC.to(DEVICE)

        self.opt_gen = optim.Adam(self.GEN.parameters(), lr=self.lr, betas=(b1, b2))
        self.opt_critic = optim.Adam(self.CRITIC.parameters(), lr=self.lr, betas=(b1, b2))
        #self.criterion = nn.BCELoss() ## not needed for wgan

        self.state = "ready"

    def process_batch(self, value, labels):
        x = value.to(DEVICE) # real
        for i in range(self.critic_iters):
            ## Get a batch of noise and run it through G to get a fake batch
            z = torch.randn(value.shape[0], self.z_size, 1, 1).to(DEVICE)
            g_z = self.GEN(z) ## fake
            ## Run the real batch through D and compute D's real loss
            d_x = self.CRITIC(x) ## critic real
            #loss_d_x = self.criterion(d_x, torch.ones_like(d_x)) ## not needed for wgan
            ## Run the fake batch through D and compute D's fake loss
            d_g_z = self.CRITIC(g_z.detach()) ## critic fake
            #loss_d_g_z = self.criterion(d_g_z, torch.zeros_like(d_g_z)) ## not needed for wgan
            gp = gradient_penalty(self.CRITIC, x, g_z, device=DEVICE)
            loss_critic = (
                -(torch.mean(d_x) - torch.mean(d_g_z)) + self.lambda_gp * gp
            )
            self.CRITIC.zero_grad()
            loss_critic.backward(retain_graph=True)
            self.opt_critic.step()

        ## Get D's total loss
        #loss_d = (loss_d_x + loss_d_g_z)/2 ## not needed for wgan
        ## Train D
        #self.DISC.zero_grad()
        #loss_d.backward()
        #self.opt_disc.step()
        ## Rerun the fake batch through trained D, then train G
        output = self.CRITIC(g_z) # critic fake post training
        loss_g = -torch.mean(output)
        self.GEN.zero_grad()
        loss_g.backward()
        self.opt_gen.step()

    def get_fixed(self):
        return torch.randn(self.batch_size, self.z_size, 1, 1).to(DEVICE)

    def serialize(self):
        return {
            'G_state': self.GEN.state_dict(),
            'D_state': self.CRITIC.state_dict(),
            'optG_state': self.opt_gen.state_dict(),
            'optD_state':self.opt_critic.state_dict(),
        }

    def deserialize(self, state):
        self.GEN.load_state_dict(state['G_state'])
        self.CRITIC.load_state_dict(state['D_state'])
        self.opt_gen.load_state_dict(state['optG_state'])
        self.opt_critic.load_state_dict(state['optD_state'])
