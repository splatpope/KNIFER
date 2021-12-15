import math
from typing import OrderedDict
import torch
import torch.nn as nn
from ..common import UpSample, DownSample

def _init_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

## A list of integer grid sizes is valid if they are all powers of 2 and are monotonically arranged
def _validate_grids(grids):
    assert(isinstance(grids, list))
    assert(all([isinstance(g, int) for g in grids]))
    if any([not math.log2(g).is_integer() for g in grids]):
        return False
    if grids == sorted(grids) or grids == sorted(grids, reverse=True):
        return True
    return False

## Compute grid transitions to get layer parameters
def _layers(grids):
    transitions = []
    if grids[0] < grids[1]: # upscaling
        for i in range(len(grids)-1):
            transitions.append(int(grids[i+1]/grids[i]))
    else: # downscaling
        for i in range(len(grids)-1):
            transitions.append(int(grids[i]/grids[i+1]))
    return transitions

class Generator(nn.Module):
    def __init__(self, grids, out_channels, latent_size, n_gpu=1, features=None, feature_scales=None):
        super(Generator, self).__init__()
        
        _validate_grids(grids)
        assert(grids[0] == 1)   ## input must be 1-D
        if not features:    ## output_img size by default
            features = grids[-1]
        upscales = _layers(grids)
        if not feature_scales:  ## divide by 2 each step by default
            feature_scales = [2**-i for i in range(len(upscales)-1)]

        self.n_f = features
        self.n_c = out_channels
        self.n_z = latent_size
        self.n_gpu = n_gpu
        ## Create input layer
        blocks = [("input", self._input(self.n_f, upscales[0]))]
        ## Create inner layers
        for i, factor in enumerate(upscales[1:-1]):
            f_in = self.n_f * feature_scales[i]
            f_out = self.n_f * feature_scales[i+1]
            blocks.append(
                (f"up_{i}", self._inner_block(int(f_in), int(f_out), factor))
            )
        ## Create output layer
        last_f = self.n_f * feature_scales[-1]
        blocks.append(
            ("output", self._output(int(last_f), upscales[-1]))
        )
        ## Compose
        self.main = nn.Sequential(OrderedDict(blocks))

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

    def forward(self, input):
        if not self.main:
            raise ValueError
        if input.is_cuda and self.n_gpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.n_gpu))
        else:
            output = self.main(input)
        return output

class Discriminator(nn.Module):
    def __init__(self, grids, channels, leak_f=0.2, n_gpu=1, features=None, feature_scales=None):
        super(Discriminator, self).__init__()

        _validate_grids(grids)
        assert(grids[-1] == 1)  ## output must be 1-D
        if not features:
            features = grids[0]
        downscales = _layers(grids)
        if not feature_scales:  ## multiply by 2 each step by default
            feature_scales = [2**i for i in range(len(downscales)-1)]

        self.n_f = features
        self.n_c = channels
        self.leak_f = leak_f
        self.n_gpu = n_gpu
        ## Create input layer
        blocks = [("input", self._input(self.n_f, downscales[0]))]
        ## Create inner layers
        for i, factor in enumerate(downscales[1:-1]):
            f_in = self.n_f * feature_scales[i]
            f_out = self.n_f * feature_scales[i+1]
            blocks.append(
                (f"down_{i}", self._inner_block(int(f_in), int(f_out), factor))
            )
        ## Create output layer
        last_f = self.n_f * feature_scales[-1]
        blocks.append(
            ("output", self._output(int(last_f), downscales[-1]))
        )
        ## Compose
        self.main = nn.Sequential(OrderedDict(blocks))

        _init_weights(self)

    def _input(self, features, start_grid):
        return nn.Sequential(
            DownSample(self.n_c, features, start_grid),
            nn.LeakyReLU(True),
        )

    def _inner_block(self, in_c, out_c, factor):
        return nn.Sequential(
            DownSample(in_c, out_c, factor, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(True),
        )

    def _output(self, features, factor):
        return nn.Sequential(
            nn.Conv2d(features, 1, factor),
            nn.Sigmoid(),
        )

    def forward(self, input):
        if not self.main:
            raise ValueError
        if input.is_cuda and self.n_gpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.n_gpu))
        else:
            output = self.main(input)
        return output.view(-1, 1).squeeze(1)

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