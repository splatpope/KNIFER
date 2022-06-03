import torch.nn as nn

from . layers import *
from ..common import BaseDiscriminator, BaseGenerator

# attn_spots is a list of booleans telling if the layer at same index
# should be followed by a self attention module
# please dont put self attn after first or last layers
class Generator(BaseGenerator):
    def __init__(self, params):
        super().__init__(params)
        self.n_c = params["img_channels"]
        self.n_z = params["latent_size"]
        self.upscales = params["upscales"]
        self.features = params["features_g"]
        self.attn_spots = params["attn_spots"]

        self.in_layer = GenInputLayer(self.n_z, self.features[0])
        self.mid_layers = nn.ModuleList([
            GenMidLayer(
                self.features[i], 
                self.features[i+1], 
                self.upscales[i],
                attn=i in self.attn_spots,
            ) for i,_ in enumerate(self.features[0:-1])
        ])
        self.out_layer = GenOutputLayer(self.features[-1], self.n_c, self.upscales[-1])

    def forward(self, z):
        out = self.in_layer(z)
        for ml in self.mid_layers:
            out = ml(out)
        out = self.out_layer(out)
        return out

    @classmethod
    def get_required_params(cls):
        return super().get_required_params() + [
            "attn_spots"
        ]

class Discriminator(BaseDiscriminator):
    def __init__(self, params, leak=0.2):
        super().__init__(params)
        self.n_c = params["img_channels"]
        self.downscales = params["downscales"]
        self.features = params["features_d"]
        self.attn_spots = params["attn_spots"]

        self.in_layer = DiscInputLayer(self.n_c, self.features[0], self.downscales[0], leak)
        self.mid_layers = nn.ModuleList([
            DiscMidLayer(
                self.features[i], 
                self.features[i+1], 
                self.downscales[i+1],
                attn=i in self.attn_spots,
            ) for i,_ in enumerate(self.features[0:-1])
        ])
        self.out_layer = DiscOutputLayer(self.features[-1])

    def forward(self, img):
        out = self.in_layer(img)
        for ml in self.mid_layers:
            out = ml(out)
        out = self.out_layer(out)
        return out.view(-1, 1).squeeze(1)
    
    @classmethod
    def get_required_params(cls):
        return super().get_required_params() + [
            "attn_spots"
        ]
