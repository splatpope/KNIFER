import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from .model import Generator, Discriminator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer():
    def __init__(self, dataset, batch_size=16, learning_rate=2e-4, latent_size=100):
        self.batch_size = batch_size
        self.lr = learning_rate
        self.z_size = latent_size

        sample = dataset[0][0]
        self.img_size = sample.shape[2]
        self.channels = sample.shape[0]

        self.data = DataLoader(dataset, self.batch_size, shuffle=True)

        self.GEN = Generator(self.img_size, self.channels, self.z_size)
        self.DISC = Discriminator(self.img_size, self.channels)

        self.opt_gen = optim.Adam(self.GEN.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.opt_disc = optim.Adam(self.DISC.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.criterion = nn.BCELoss()

    def process_batch(self, value, id):
        print(f"batch {id} of {len(self.data)}")
        x = value.to(DEVICE)
        z = torch.randn(self.batch_size, self.z_size, 1, 1).to(DEVICE)
        g_z = self.GEN(z)

        d_x = self.DISC(x)
        loss_d_x = self.criterion(d_x, torch.ones_like(d_x))

        d_g_z = self.DISC(g_z.detach())
        loss_d_g_z = self.criterion(d_g_z, torch.zeros_like(d_g_z))

        loss_d = (loss_d_x + loss_d_g_z)/2

        self.DISC.zero_grad()
        loss_d.backward()
        self.opt_disc.step()

        output = self.DISC(g_z)
        loss_g = self.criterion(output, torch.ones_like(output))
        self.GEN.zero_grad()
        loss_g.backward()
        self.opt_gen.step()

    def train(self, n_iter):
        fixed = torch.randn(self.batch_size, self.z_size, 1, 1).to(DEVICE)
        for epoch in range(n_iter):
            print(f"epoch {epoch} of {n_iter}")
            for batch_id, (real, _) in enumerate(self.data):
                
            with torch.no_grad():
                fake = self.GEN(fixed).detach()
                vutils.save_image(fake, "./out.png", normalize=True)
