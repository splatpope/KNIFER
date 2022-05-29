import math
from typing import OrderedDict
import torch
import torch.nn as nn
from ..common import UpSample, DownSample, validate_grids, layers
import architectures.dcgan.model as DCGAN

def _init_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

class Generator(DCGAN.Generator):
    def __init__(self, params, n_gpu=1, features=None, feature_scales=None):
        super(Generator, self).__init__(params, n_gpu, features, feature_scales)
        _init_weights(self)

    def _input(self, features, start_grid):
        return nn.Sequential(
            nn.ConvTranspose2d(self.n_z, features, start_grid, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
        )

    def _inner_block(self, in_c, out_c, factor):
        return nn.Sequential(
            UpSample(in_c, out_c, factor, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True),
        )

    def _output(self, features, factor):
        return nn.Sequential(
            UpSample(features, self.n_c, factor),
            nn.Tanh(),
        )

class Discriminator(DCGAN.Discriminator):
    def __init__(self, params, leak_f=0.2, n_gpu=1, features=None, feature_scales=None):
        super(Discriminator, self).__init__(params, leak_f, n_gpu, features, feature_scales)
        _init_weights(self)

    def _input(self, features, start_grid):
        return nn.Sequential(
            DownSample(self.n_c, features, start_grid),
            nn.LeakyReLU(self.leak_f, True),
        )

    def _inner_block(self, in_c, out_c, factor):
        return nn.Sequential(
            DownSample(in_c, out_c, factor, bias=False),
            nn.InstanceNorm2d(out_c, affine=True),
            nn.LeakyReLU(self.leak_f, True),
        )

    def _output(self, features, factor):
        return nn.Sequential(
            nn.Conv2d(features, 1, factor),
        )

#TODO : OUTDATED
def test(batch_size=16, latent_size=100, img_size=64, channels=3):
    device = torch.device("cpu")
    z = torch.randn(batch_size, latent_size, 1, 1).to(device)
    x = torch.randn(batch_size, channels, img_size, img_size).to(device)
    gen_grids = [1, 4, 32, 256]
    disc_grids = gen_grids[::-1]

    GEN = Generator(gen_grids, channels, latent_size, features=512)
    DISC = Discriminator(disc_grids, channels, features=64)

    print(GEN)
    print(DISC)
    print("noise shape", z.shape)
    print("real shape", x.shape)

    g_z = GEN(z).to(device)
    print("gen shape", g_z.shape)

    d_x = DISC(x).to(device)
    print("disc shape", d_x.shape)

if __name__ == "__main__":
    test()
