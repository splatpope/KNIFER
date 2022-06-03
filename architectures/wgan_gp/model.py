import torch.nn as nn

from . layers import *
from .. common import BaseDiscriminator, BaseGenerator

class Generator(BaseGenerator):
    def __init__(self, params: dict):
        super().__init__(params)

        self.n_c = params["img_channels"]
        self.n_z = params["latent_size"]
        self.upscales = params["upscales"]
        self.features = params["features_g"]

        self.in_layer = GenInputLayer(self.n_z, self.features[0])
        self.mid_layers = nn.ModuleList([
            GenMidLayer(
                self.features[i], 
                self.features[i+1], 
                self.upscales[i]
            ) for i,_ in enumerate(self.features[0:-1])
        ])
        self.out_layer = GenOutputLayer(self.features[-1], self.n_c, self.upscales[-1])

    def forward(self, z):
        out = self.in_layer(z)
        for ml in self.mid_layers:
            out = ml(out)
        out = self.out_layer(out)
        return out

class Discriminator(BaseDiscriminator):
    def __init__(self, params, leak=0.2):
        super().__init__(params)

        self.n_c = params["img_channels"]
        self.downscales = params["downscales"]
        self.features = params["features_d"]

        self.in_layer = DiscInputLayer(self.n_c, self.features[0], self.downscales[0], leak)
        self.mid_layers = nn.ModuleList([
            DiscMidLayer(
                self.features[i], 
                self.features[i+1], 
                self.downscales[i+1]
            ) for i,_ in enumerate(self.features[0:-1])
        ])
        self.out_layer = DiscOutputLayer(self.features[-1])

    def forward(self, img):
        out = self.in_layer(img)
        for ml in self.mid_layers:
            out = ml(out)
        out = self.out_layer(out)
        return out.view(-1, 1).squeeze(1)
