from .dcgan.train import Trainer as DCGANTrainer
from .wgan_gp.train import Trainer as WGAN_GPTrainer
from .sagan.train import Trainer as SAGANTrainer
from .sagan.train import WGPTrainer as SAGANWGPTrainer
## Custom models for 256 img size with 3 layers
def DCGANTrainer_256_3(dset, params):
    params.update({
        'grids_g': [1, 4, 32, 256],
        'grids_d': [256, 32, 4, 1],
    })
    return DCGANTrainer(dset, params) 

def DCGANTrainerEZMnist(dset, params):
    params.update({
        'grids_g': [1, 16, 32],
        'grids_d': [32, 16, 1],
    })
    return DCGANTrainer(dset, params)

def WGAN_GPTrainer_256_3(dset, params):
    params.update({
        'grids_g': [1, 4, 32, 256],
        'grids_d': [256, 32, 4, 1],
    })
    return WGAN_GPTrainer(dset, params) 

def SAGANTrainer_32_4_2a(dset, params):
    params.update({
        'grids_g': [1, 4, 8, 16, 32],
        'grids_d': [32, 16, 8, 4, 1],
        'attn_spots' : [0, 1]
    })
    return SAGANTrainer(dset, params)

def SAGANTrainer_32_4_2a_WGP(dset, params):
    params.update({
        'grids_g': [1, 4, 8, 16, 32],
        'grids_d': [32, 16, 8, 4, 1],
        'attn_spots' : [0, 1]
    })
    return SAGANWGPTrainer(dset, params)

def SAGANTrainer_32_4_2a_4f(dset, params):
    params.update({
        
    })