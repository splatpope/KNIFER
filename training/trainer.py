from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from . loss import BaseLoss, BCELoss, WGPLoss, HingeLoss

from architectures import *

from misc_utils import set_req_grads

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

ARCH2MODS = {
    "DCGAN": (DCGen, DCDisc),
    "WGAN": (WGen, WDisc),
    "SAGAN": (SAGen, SADisc),
}

LOSS_CLASSES = {
    "DC": BCELoss,
    "BCE": BCELoss,
    "WGP": WGPLoss,
    "HINGE": HingeLoss,
}

DEFAULT_LOSSES = {
    "DCGAN": "BCE",
    "WGAN": "WGP",
    "SAGAN": "HINGE",
}

class GANTrainer():
    def __init__(self, dataset, params: dict, num_workers=0):
        self.arch = params["arch"]
        self.batch_size = params["batch_size"]
        self.latent_size = params["latent_size"]
        self.lr_g = params["lr_g"]
        self.lr_d = params["lr_d"]
        self.b1 = params["b1"]
        self.b2 = params["b2"]

        if "features" in params:
            self.features = params["features"]
        else:
            self.features = None

        if not "loss" in params:
            self.loss = DEFAULT_LOSSES[self.arch]
            params["loss"] = self.loss
        else:
            self.loss =  params["loss"]

        sample = dataset[0][0]

        self.img_size = sample.shape[2]
        if not "img_size" in params: ## should never happen but hey
            params["img_size"] = self.img_size

        self.channels = sample.shape[0]
        if not "img_channels" in params:
            params["img_channels"] = self.channels

        pin_memory = torch.cuda.is_available()

        self.data = DataLoader(dataset, self.batch_size, 
            shuffle=True, 
            pin_memory=pin_memory, 
            num_workers=num_workers,
        )

        self.use_SR = params["use_sr"]
        self.highest_sigmas = None

        self.build(params)

    def build(self, params):
        # Discover proper arch models, instanciate them, then transfer to device
        gen_mod, disc_mod = ARCH2MODS[self.arch]
        self.GEN: nn.Module = gen_mod(params)
        self.DISC: nn.Module = disc_mod(params)
        self.GEN.to(DEVICE)
        self.DISC.to(DEVICE)

        # Setup optimizers
        betas = (self.b1, self.b2)
        self.opt_gen = optim.Adam(self.GEN.parameters(), lr=self.lr_g, betas=betas)
        self.opt_disc = optim.Adam(self.DISC.parameters(), lr=self.lr_d, betas=betas)
        
        # Create loss class
        crit_class = LOSS_CLASSES[self.loss]
        if type(crit_class) == WGPLoss:
            self.criterion = crit_class(self.DISC, device=DEVICE) # thanks wgp
        else:
            self.criterion = crit_class(device=DEVICE)

    def process_batch(self, value, labels):
        ## Reset gradients
        self.opt_gen.zero_grad()
        self.opt_disc.zero_grad()

        ## Train D
        # Disable G's gradients
        set_req_grads(self.GEN, False)
        set_req_grads(self.DISC, True)
        # Compute D's output w.r.t real image
        real = value.to(DEVICE)
        D_real = self.DISC(real)
        # Generate detached fake image from latent vector
        z = torch.randn(self.batch_size, self.latent_size, 1, 1, device=DEVICE)
        fake = self.GEN(z).detach()
        # Compute D's output w.r.t fake image
        D_fake = self.DISC(fake)
        # Compute D's losses
        loss_D, loss_D_real_val, loss_D_fake_val = self.criterion.D(D_real, D_fake)
        # Backward and step D
        loss_D.backward()
        self.opt_disc.step()

        if self.use_SR:
            self.spectral_regularization()

        self.opt_gen.zero_grad() ## Just in case

        ## Train G
        # Disable D's gradients
        set_req_grads(self.GEN, True)
        set_req_grads(self.DISC, False)
        # Generate another fake image from another latent vector
        z = torch.randn(self.batch_size, self.latent_size, 1, 1, device=DEVICE)
        fake = self.GEN(z)
        # Compute D's output w.r.t this new fake image
        D_fake = self.DISC(fake)
        # Compute G's loss
        loss_G = self.criterion.G(D_fake)
        # Backward and step G
        loss_G.backward()
        self.opt_gen.step()



        # Produce loss report
        return loss_G.item(), loss_D.item(), loss_D_real_val, loss_D_fake_val

    def spectral_regularization(self):
        with torch.no_grad():
            if self.highest_sigmas is None:
                self.highest_sigmas = {}
                for name, m in self.DISC.named_modules():
                    if isinstance(m, nn.Conv2d):
                        W = m.weight.data.reshape(m.weight.shape[0], -1).to('cpu')
                        s = torch.linalg.svdvals(W)
                        self.highest_sigmas[name] = s/max(s)
            else:
                for name, m in self.DISC.named_modules():
                    if isinstance(m, nn.Conv2d):
                        W = m.weight.data.reshape(m.weight.shape[0], -1).to('cpu')
                        U, s, V = torch.linalg.svd(W, full_matrices=False)
                        s1 = max(s)
                        s = s / s1
                        self.highest_sigmas[name] = torch.maximum(s, self.highest_sigmas[name])
                        new_s = s1 * self.highest_sigmas[name]
                        S = torch.diag(new_s)
                        W = U @ S @ V
                        m.weight.data = W.reshape(m.weight.shape).to(DEVICE)

    @classmethod
    def get_required_params(cls):
        return [
            "arch",
            "batch_size",
            "latent_size",
            "lr_g",
            "lr_d",
            "b1",
            "b2",
            "upscales",
            "downscales",
            "features_g",
            "features_d",
        ]
    
    def get_fixed(self):
        return torch.randn(self.batch_size, self.latent_size, 1, 1, device=DEVICE)

    def parallelize(self):
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                self.GEN = nn.DataParallel(self.GEN)
                self.DISC = nn.DataParallel(self.DISC)

    def serialize(self):
        return {
            'G_state': self.GEN.state_dict(),
            'D_state': self.DISC.state_dict(),
            'optG_state': self.opt_gen.state_dict(),
            'optD_state':self.opt_disc.state_dict(),
        }

    def deserialize(self, state):
        self.GEN.load_state_dict(state['G_state'])
        self.DISC.load_state_dict(state['D_state'])
        self.opt_gen.load_state_dict(state['optG_state'])
        self.opt_disc.load_state_dict(state['optD_state'])
