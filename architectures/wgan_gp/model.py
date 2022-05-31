import math
from typing import OrderedDict
import torch
import torch.nn as nn
from ..common import UpSample, DownSample, validate_grids, layers
import architectures.dcgan.model as DCGAN

class Generator(DCGAN.Generator):
    def __init__(self, params, n_gpu=1, features=None, feature_scales=None):
        super(Generator, self).__init__(params, n_gpu=n_gpu, features=features, feature_scales=feature_scales)

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
        super(Discriminator, self).__init__(params, leak_f=leak_f, n_gpu=n_gpu, features=features, feature_scales=feature_scales)

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
