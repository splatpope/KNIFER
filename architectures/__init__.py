from .dcgan.train import Trainer as DCGANTrainer
from .wgan_gp.train import Trainer as WGAN_GPTrainer
## Custom models for 256 img size with 3 layers
def DCGANTrainer_256_3(dset, params):
    params.update({
        'grids_g': [1, 4, 32, 256],
        'grids_d': [256, 32, 4, 1],
    })
    return DCGANTrainer(dset, params) 

def WGAN_GPTrainer_256_3(dset, params):
    params.update({
        'grids_g': [1, 4, 32, 256],
        'grids_d': [256, 32, 4, 1],
    })
    return WGAN_GPTrainer(dset, params) 
