from functools import partial
import contextlib

import torch
import torch.nn as nn
import torch.optim as optim

from knifer.config import params as P
from knifer.misc_utils import set_req_grads

from . loss import STR_TO_LOSS, BaseGANLoss

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# TODO: profiler
class GANUpdater():
    def __init__(self, params: P.UpdaterParameters):
        import knifer.context as KF

        self.batch_size = params.batch_size
        self.criterion: BaseGANLoss = STR_TO_LOSS[params.loss](device=DEVICE)
        self.regularization = params.regularization
        self.opt_G = params.opt_G(KF.ARCH.gen)
        self.opt_D = params.opt_D(KF.ARCH.disc)

    def process_batch(self, value:torch.Tensor, labels: torch.Tensor):
        import knifer.context as KF
        G = KF.ARCH.gen
        D = KF.ARCH.disc
        z_shape = torch.Size((self.batch_size, KF.ARCH.latent_size, 1, 1))

        ## Reset gradients
        self.opt_G.zero_grad()
        self.opt_D.zero_grad()

        ## Train D
        # Disable G's gradients
        set_req_grads(G, False)
        set_req_grads(D, True)
        # Compute D's output w.r.t real image
        real = value.to(DEVICE)
        D_real = D(real)
        # Generate detached fake image from latent vector
        z = torch.randn(z_shape, device=DEVICE)
        fake = G(z).detach()
        # Compute D's output w.r.t fake image
        D_fake = D(fake)
        # Compute D's losses
        loss_D, loss_D_real_val, loss_D_fake_val = self.criterion.D(D_real, D_fake)
        # Backward and step D
        loss_D.backward()
        self.opt_D.step()

        # self.regularization.step()

        self.opt_G.zero_grad() ## Just in case

        ## Train G
        # Disable D's gradients
        set_req_grads(G, True)
        set_req_grads(D, False)
        # Generate another fake image from another latent vector
        z = torch.randn(z_shape, device=DEVICE)
        fake = G(z)
        # Compute D's output w.r.t this new fake image
        D_fake = D(fake)
        # Compute G's loss
        loss_G = self.criterion.G(D_fake)
        # Backward and step G
        loss_G.backward()
        self.opt_G.step()

        # Produce loss report
        losses_report = [loss_G.item(), loss_D.item(), loss_D_real_val, loss_D_fake_val]
        KF.LOGGER.update_stats(losses_report)

    def serialize(self):
        return {
            'optG_state': self.opt_G.state_dict(),
            'optD_state':self.opt_D.state_dict(),
        }

    def deserialize(self, state):
        self.opt_G.load_state_dict(state['optG_state'])
        self.opt_D.load_state_dict(state['optD_state'])
