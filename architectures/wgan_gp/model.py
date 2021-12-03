import math
from typing import OrderedDict
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, img_size, channels, latentSize, n_gpu=1, features=None):
        super(Generator, self).__init__()
        self.img_size = img_size
        if not features:
            features = self.img_size
        self.n_f = features
        self.n_c = channels
        self.n_z = latentSize
        self.n_gpu = n_gpu
        self.main = self._compose()
        _init_weights(self)

    def _compose(self):
        n_b = _n_inner_blocks(self.img_size)
        ff = int(2**n_b)
        ops = list()
        ## Create input block
        ops += self._block(self.n_z, self.n_f * ff, 4, 1, 0)
        ## Calculate middle blocks
        while(ff > 1):
            in_c = int(self.n_f * ff)
            ff /= 2
            out_c = int(self.n_f * ff)
            ops += self._block(in_c, out_c, 4, 2, 1)
        ## Create output block
        ops += [
            (f"up_{int(ff)}", nn.ConvTranspose2d(self.n_f, self.n_c, kernel_size=4, stride=2, padding=1, bias=False)),
            ("img_out", nn.Tanh()),
        ]
        return nn.Sequential(OrderedDict(ops))

    def _block(self, in_c, out_c, kernel_size, stride, padding):
        return [
            (f"up_{out_c}", nn.ConvTranspose2d(in_c, out_c, kernel_size, stride, padding, bias=False)),
            (f"bn_{out_c}" ,nn.BatchNorm2d(out_c)),
            (f"relu_{out_c}", nn.ReLU(True)),
        ]

    def forward(self, input):
        if input.is_cuda and self.n_gpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.n_gpu))
        else:
            output = self.main(input)
        return output

class Discriminator(nn.Module):
    def __init__(self, img_size, channels, leak_f=0.2, n_gpu=1, features=None):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        if not features:
            features = self.img_size
        self.n_f = features
        self.n_c = channels
        self.leak_f = leak_f
        self.n_gpu = n_gpu
        self.main = self._compose()
        _init_weights(self)

    def _compose(self):
        n_b = _n_inner_blocks(self.img_size)
        ff = 1
        ops = list()
        ## Create input block
        ops += [
            (f"down_{int(ff)}", nn.Conv2d(self.n_c, self.n_f * ff, 4, 2, 1)),
            (f"lrelu_{int(ff)}", nn.LeakyReLU(self.leak_f, True)),
        ]
        ## Calculate middle blocks
        while(ff < int(2**n_b)):
            in_c = int(self.n_f * ff)
            ff *= 2
            out_c = int(self.n_f * ff)
            ops += self._block(in_c, out_c, 4, 2, 1)
        ## Create output block
        ops += [
            (f"down_{int(ff)}", nn.Conv2d(self.n_f * ff, 1, kernel_size=4, stride=1, padding=0, bias=False)),
            #(f"label", nn.Sigmoid()), ## Not needed for WGAN
        ]
        return nn.Sequential(OrderedDict(ops))

    def _block(self, in_c, out_c, kernel_size, stride, padding, leak_f=0.2):
        return [
            (f"down_{out_c}", nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False)),
            (f"in_{out_c}", nn.InstanceNorm2d(out_c, affine=True)),
            (f"lrelu_{out_c}", nn.LeakyReLU(leak_f, True)),
        ]

    def forward(self, input):
        if input.is_cuda and self.n_gpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.n_gpu))
        else:
            output = self.main(input)
        return output.view(-1, 1).squeeze(1)

def _n_inner_blocks(image_size):
    return int(math.log2(image_size)) - 3

def _init_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def test(batch_size=16, latent_size=100, img_size=64, channels=3):
    device = torch.device("cpu")
    z = torch.randn(batch_size, latent_size, 1, 1).to(device)
    x = torch.randn(batch_size, channels, img_size, img_size).to(device)
    GEN = Generator(img_size, channels, latent_size, features=8)
    DISC = Discriminator(img_size, channels, features=8)

    print(GEN)
    print(DISC)
    print("noise shape", z.shape)
    print("real shape", x.shape)

    g_z = GEN(z).to(device)
    print("gen shape", g_z.shape)

    d_x = DISC(x).to(device)
    print("disc shape", d_x.shape)
