import math
from typing import OrderedDict
import torch
import torch.nn as nn
from ..common import UpSample, DownSample, validate_grids, layers, check_required_params

def _init_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

class Generator(nn.Module):
    def __init__(self, params, n_gpu=1, features=None, feature_scales=None):
        super(Generator, self).__init__()
        check_required_params(self, params)
        
        grids = params["grids_g"]
        assert(grids[0] == 1)   ## input must be 1-D
        if not features:    ## output_img size by default
            features = grids[-1] * 2**(len(grids)-3)
        upscales = layers(grids)
        if not feature_scales:  ## divide by 2 each step by default
            feature_scales = [2**-i for i in range(len(upscales)-1)]

        self.n_f = features
        self.n_c = params["img_channels"]
        self.n_z = params["latent_size"]
        self.n_gpu = n_gpu

        ## Create input layer
        #blocks = [("input", self._input(self.n_f, upscales[0]))]
        self.in_layer = self._input(self.n_f, upscales[0])
        self.mid_layers = []
        ## Create inner layers
        for i, factor in enumerate(upscales[1:-1]):
            f_in = self.n_f * feature_scales[i]
            f_out = self.n_f * feature_scales[i+1]

            #blocks.append(
            #    (f"up_{i}", self._inner_block(int(f_in), int(f_out), factor))
            #)
            self.mid_layers.append(
                self._inner_block(int(f_in), int(f_out), factor)
            )
        self.mid_layers = nn.ModuleList(self.mid_layers)
        ## Create output layer
        last_f = self.n_f * feature_scales[-1]
        #blocks.append(
        #    ("output", self._output(int(last_f), upscales[-1]))
        #)
        self.out_layer = self._output(int(last_f), upscales[-1])
        ## Compose
        #self.main = nn.Sequential(OrderedDict(blocks))

        _init_weights(self)

    def main(self, input):
        out = self.in_layer(input)
        for l in self.mid_layers:
            out = l(out)
        return self.out_layer(out)

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
        #if input.is_cuda and self.n_gpu > 1:
        #    output = nn.parallel.data_parallel(self.main, input, range(self.n_gpu))
        else:
            output = self.main(input)
        return output

    @classmethod
    def get_required_params(cls):
        return [
            "grids_g",
            "img_channels",
            "latent_size",
        ]

class Discriminator(nn.Module):
    def __init__(self, params, leak_f=0.2, n_gpu=1, features=None, feature_scales=None):
        super(Discriminator, self).__init__()
        check_required_params(self, params)

        grids = params["grids_d"]
        assert(grids[-1] == 1)  ## output must be 1-D
        if not features:
            features = grids[0]
        downscales = layers(grids)
        if not feature_scales:  ## multiply by 2 each step by default
            feature_scales = [2**i for i in range(len(downscales)-1)]

        self.n_f = features
        self.n_c = params["img_channels"]
        self.leak_f = leak_f
        self.n_gpu = n_gpu
        ## Create input layer
        #blocks = [("input", self._input(self.n_f, downscales[0]))]
        self.in_layer = self._input(self.n_f, downscales[0])
        ## Create inner layers
        self.mid_layers = []
        for i, factor in enumerate(downscales[1:-1]):
            f_in = self.n_f * feature_scales[i]
            f_out = self.n_f * feature_scales[i+1]
            #blocks.append(
            #    (f"down_{i}", self._inner_block(int(f_in), int(f_out), factor))
            #)
            self.mid_layers.append(self._inner_block(int(f_in), int(f_out), factor))
        self.mid_layers = nn.ModuleList(self.mid_layers)
        ## Create output layer
        last_f = self.n_f * feature_scales[-1]
        #blocks.append(
        #    ("output", self._output(int(last_f), downscales[-1]))
        #)
        self.out_layer = self._output(int(last_f), downscales[-1])
        ## Compose
        #self.main = nn.Sequential(OrderedDict(blocks))

        _init_weights(self)

    def main(self, input):
        out = self.in_layer(input)
        for l in self.mid_layers:
            out = l(out)
        return self.out_layer(out)

    def _input(self, features, start_grid):
        return nn.Sequential(
            DownSample(self.n_c, features, start_grid),
            nn.LeakyReLU(self.leak_f, True),
        )

    def _inner_block(self, in_c, out_c, factor):
        return nn.Sequential(
            DownSample(in_c, out_c, factor, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(self.leak_f, True),
        )

    def _output(self, features, factor):
        return nn.Sequential(
            nn.Conv2d(features, 1, factor),
            nn.Sigmoid(),
        )

    def forward(self, input):
        if not self.main:
            raise ValueError
        #if input.is_cuda and self.n_gpu > 1:
        #    output = nn.parallel.data_parallel(self.main, input, range(self.n_gpu))
        else:
            output = self.main(input)
        return output.view(-1, 1).squeeze(1)

    @classmethod
    def get_required_params(cls):
        return [
            "grids_d",
            "img_channels",
        ]

#TODO : outdated
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
