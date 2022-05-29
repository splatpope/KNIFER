import math
from typing import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import UpSample, DownSample
import architectures.dcgan.model as DCGAN
from torch_utils.spectral import SpectralNorm

# TODO : check if we need to init self attn params the same way
def _init_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

# from "official" SAGAN implementation
class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out,attention

# attn_spots is a list of indices fitting inside the mid_layers list
# which tell where to place attn layers AFTER the corresponding mid layer
class Generator(DCGAN.Generator):
    def __init__(self, params, n_gpu=1, features=None, feature_scales=None):
        super(Generator, self).__init__(params, n_gpu, features, feature_scales)
        for a in params["attn_spots"]:
            assert a < len(self.mid_layers), "Not enough mid layers for the chosen attention layer spot."
        self.attn_spots = params["attn_spots"]
        self.attn_layers = {}
        for attn in self.attn_spots:
            # we want the out channels of the conv2d module stored in a specific mid layer
            self.attn_layers[str(attn)] = (Self_Attn(self.mid_layers[attn][0].out_channels, 'relu'))
        self.attn_layers = nn.ModuleDict(self.attn_layers)

    def _input(self, features, start_grid):
        return nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(self.n_z, features, start_grid, bias=False)),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
        )
    def _inner_block(self, in_c, out_c, factor):
        return nn.Sequential(
            UpSample(in_c, out_c, factor, bias=False, spectral_norm=True),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True),
        )

    def _output(self, features, factor):
        return nn.Sequential(
            UpSample(features, self.n_c, factor, spectral_norm=True),
            nn.Tanh(),
        )
    
    def main(self, input):
        #input = input.view(input.size(0), input.size(1), 1, 1)
        out = self.in_layer(input)
        for idx, l in enumerate(self.mid_layers):
            out = l(out)
            if idx in self.attn_spots:
                out, _ = self.attn_layers[str(idx)](out)
        return self.out_layer(out)

    @classmethod
    def get_required_params(cls):
        return super().get_required_params() + [
            "attn_spots"
        ]

class Discriminator(DCGAN.Discriminator):
    def __init__(self, params, n_gpu=1, features=None, feature_scales=None):
        super(Discriminator, self).__init__(params, n_gpu, features, feature_scales)
        for a in params["attn_spots"]:
            assert a < len(self.mid_layers), "Not enough mid layers for the chosen attention layer spot."
        self.attn_spots = params["attn_spots"]
        self.attn_layers = {}
        for attn in self.attn_spots:
            # we want the out channels of the conv2d module stored in a specific mid layer
            self.attn_layers[str(attn)] = (Self_Attn(self.mid_layers[attn][0].out_channels, 'relu'))
        self.attn_layers = nn.ModuleDict(self.attn_layers)
    
    def main(self, input):
        #input = input.view(input.size(0), input.size(1), 1, 1)
        out = self.in_layer(input)
        for idx, l in enumerate(self.mid_layers):
            out = l(out)
            if idx in self.attn_spots:
                out, _ = self.attn_layers[str(idx)](out)
        return self.out_layer(out)

    def _input(self, features, start_grid):
        return nn.Sequential(
            DownSample(self.n_c, features, start_grid, spectral_norm=True),
            nn.LeakyReLU(self.leak_f, True),
        )

    def _inner_block(self, in_c, out_c, factor):
        return nn.Sequential(
            DownSample(in_c, out_c, factor, bias=False, spectral_norm=True),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(self.leak_f, True),
        )

    def _output(self, features, factor):
        return nn.Sequential(
            SpectralNorm(nn.Conv2d(features, 1, factor)),
            nn.Sigmoid(),
        )
    

    @classmethod
    def get_required_params(cls):
        return super().get_required_params() + [
            "attn_spots"
        ]
