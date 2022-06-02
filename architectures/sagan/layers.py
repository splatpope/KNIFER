from glob import escape
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm

from ..common import UpKConv2D, DownKConv2D

class SelfAttention(nn.Module):
    def __init__(self, in_channel, gain=1):
        super().__init__()

        self.query = spectral_norm(nn.Conv1d(in_channel, in_channel // 8, 1),
                                   n_power_iterations=gain)
        self.key = spectral_norm(nn.Conv1d(in_channel, in_channel // 8, 1),
                                 n_power_iterations=gain)
        self.value = spectral_norm(nn.Conv1d(in_channel, in_channel, 1),
                                   n_power_iterations=gain)

        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, input):
        shape = input.shape
        flatten = input.view(shape[0], shape[1], -1)
        query = self.query(flatten).permute(0, 2, 1)
        key = self.key(flatten)
        value = self.value(flatten)
        query_key = torch.bmm(query, key)
        attn = F.softmax(query_key, 1)
        attn = torch.bmm(value, attn)
        attn = attn.view(*shape)
        out = self.gamma * attn + input

        return out

def NoOp(x):
    return x

################## Generator modules ####################

#spectralnorm or not ? might wanna test, buddy

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
        out = self.act_fn(out)
        return out

class GenMidLayer(nn.Module):
    def __init__(self, in_c, out_c, factor, attn=False):
        super().__init__()
        self.conv = UpKConv2D(in_c, out_c, factor, bias=False)
        nn.init.kaiming_normal_(self.conv.weight.data)

        self.bn = nn.BatchNorm2d(out_c)
        nn.init.normal_(self.bn.weight.data, 0.0, 0.2)

        self.act_fn = nn.ReLU()

        self.attn = SelfAttention(out_c) if attn else NoOp
        

    def forward(self, input):
        out = self.conv(input)
        out = self.bn(out)
        out = self.act_fn(out)
        out = self.attn(out)
        return out

class GenOutputLayer(nn.Module):
    def __init__(self, in_c, img_channels, factor):
        super().__init__()
        self.conv = UpKConv2D(in_c, img_channels, factor)
        nn.init.xavier_normal_(self.conv.weight.data)
        nn.init.zeros_(self.conv.bias.data)

        self.act_fn = nn.Tanh()

    def forward(self, input):
        img = self.conv(input)
        img = self.act_fn(img)
        return img

################## Discriminator modules ####################

class DiscInputLayer(nn.Module):
    def __init__(self, img_channels, out_c, factor, leak=0.2, spectral=True):
        super().__init__()
        self.conv = DownKConv2D(img_channels, out_c, factor)
        if spectral:
            self.conv = spectral_norm(self.conv)
        else:
            nn.init.kaiming_uniform_(self.conv.weight.data)
            nn.init.zeros_(self.conv.bias.data)

        self.act_fn = nn.LeakyReLU(negative_slope=leak)
    
    def forward(self, img):
        out = self.conv(img)
        out = self.act_fn(out)
        return out

class DiscMidLayer(nn.Module):
    def __init__(self, in_c, out_c, factor, leak=0.2, spectral=True, attn=False):
        super().__init__()
        self.conv = DownKConv2D(in_c, out_c, factor, bias=False)
        if spectral:
            self.conv = spectral_norm(self.conv)
        else:
            nn.init.kaiming_normal_(self.conv.weight.data)

        self.bn = nn.BatchNorm2d(out_c)
        nn.init.normal_(self.bn.weight.data, 0.0, 0.2)

        self.act_fn = nn.LeakyReLU(negative_slope=leak)

        self.attn = SelfAttention(out_c) if attn else NoOp

    def forward(self, input):
        out = self.conv(input)
        out = self.bn(out)
        out = self.act_fn(out)
        out = self.attn(out)
        return out

class DiscOutputLayer(nn.Module):
    def __init__(self, in_c, spectral=True):
        super().__init__()
        self.conv = nn.Conv2d(in_c, 1, 4)
        if spectral:
            self.conv = spectral_norm(self.conv)
        else:
            nn.init.normal_(self.conv.weight.data, 0.0, 0.2) # no xavier because no sigmoid
            nn.init.zeros_(self.conv.bias.data)

    def forward(self, input):
        out = self.conv(input)
        return out
