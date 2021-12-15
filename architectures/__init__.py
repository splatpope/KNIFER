from .dcgan.train import Trainer as DCGANTrainer
def DCGANTrainer_256_3(dset, params):
    params.update({
        'grids_g': [1, 4, 32, 256],
        'grids_d': [256, 32, 4, 1],
    })
    return DCGANTrainer(dset, params) 
from .wgan_gp.train import Trainer as WGAN_GPTrainer