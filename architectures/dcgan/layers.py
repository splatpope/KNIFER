from pyexpat import features
import torch
import torch.nn as nn

from ..common import UpKConv2D, DownKConv2D

################## Generator modules ####################

class GenInputLayer(nn.Module):
    def __init__(self, latent_size, out_c):
        super().__init__()
        self.conv = nn.ConvTranspose2d(latent_size, out_c, 4, bias=False)
        nn.init.kaiming_normal_(self.conv.weight.data)

        self.bn = nn.BatchNorm2d(out_c)
        nn.init.normal_(self.bn.weight.data, 0.0, 0.2)

        self.act_fn = nn.ReLU()
    
    def forward(self, z):
        out = self.conv(z)
        out = self.bn(out)
        return self.act_fn(out)

class GenMidLayer(nn.Module):
    def __init__(self, in_c, out_c, factor):
        super().__init__()
        self.conv = UpKConv2D(in_c, out_c, factor, bias=False)
        nn.init.kaiming_normal_(self.conv.weight.data)

        self.bn = nn.BatchNorm2d(out_c)
        nn.init.normal_(self.bn.weight.data, 0.0, 0.2)

        self.act_fn = nn.ReLU()

    def forward(self, input):
        out = self.conv(input)
        out = self.bn(out)
        return self.act_fn(out)

class GenOutputLayer(nn.Module):
    def __init__(self, in_c, img_channels, factor):
        super().__init__()
        self.conv = UpKConv2D(in_c, img_channels, factor)
        nn.init.xavier_normal_(self.conv.weight.data)
        nn.init.zeros_(self.conv.bias.data)

        self.act_fn = nn.Tanh()

    def forward(self, input):
        img = self.conv(input)
        return self.act_fn(img)

################## Discriminator modules ####################

class DiscInputLayer(nn.Module):
    def __init__(self, img_channels, out_c, factor, leak=0.2):
        super().__init__()
        self.conv = DownKConv2D(img_channels, out_c, factor)
        nn.init.kaiming_uniform_(self.conv.weight.data)
        nn.init.zeros_(self.conv.bias.data)

        self.act_fn = nn.LeakyReLU(negative_slope=leak)
    
    def forward(self, img):
        out = self.conv(img)
        return self.act_fn(out)

class DiscMidLayer(nn.Module):
    def __init__(self, in_c, out_c, factor, leak=0.2):
        super().__init__()
        self.conv = DownKConv2D(in_c, out_c, factor, bias=False)
        nn.init.kaiming_normal_(self.conv.weight.data)

        self.bn = nn.BatchNorm2d(out_c)
        nn.init.normal_(self.bn.weight.data, 0.0, 0.2)

        self.act_fn = nn.LeakyReLU(negative_slope=leak)

    def forward(self, input):
        out = self.conv(input)
        out = self.bn(out)
        return self.act_fn(out)

class DiscOutputLayer(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, 1, 4)
        nn.init.xavier_normal_(self.conv.weight.data)
        nn.init.zeros_(self.conv.bias.data)
        
        self.act_fn = nn.Sigmoid()

    def forward(self, input):
        out = self.conv(input)
        return self.act_fn(out)
