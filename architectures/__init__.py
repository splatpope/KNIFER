from .dcgan.train import DC_Trainer 
from .wgan_gp.train import WGP_Trainer 
from .sagan.train import SA_Trainer, SA_WGP_Trainer

def DC_Trainer_default(params):
    return DC_Trainer
def WGP_Trainer_default(params):
    return WGP_Trainer
def SA_Trainer_default(params):
    return SA_Trainer
def SA_WGP_Trainer_default(params):
    return SA_WGP_Trainer

## Custom models for 256 img size with 3 layers
def DC_Trainer_256_3(params):
    params.update({
        'grids_g': [1, 4, 32, 256],
        'grids_d': [256, 32, 4, 1],
    })
    return DC_Trainer

def DC_Trainer_EZMnist(params):
    params.update({
        'grids_g': [1, 16, 32],
        'grids_d': [32, 16, 1],
    })
    return DC_Trainer

def WGP_Trainer_256_3(params):
    params.update({
        'grids_g': [1, 4, 32, 256],
        'grids_d': [256, 32, 4, 1],
    })
    return WGP_Trainer

def SA_Trainer_32_4_2a(params):
    params.update({
        'grids_g': [1, 4, 8, 16, 32],
        'grids_d': [32, 16, 8, 4, 1],
        'attn_spots' : [0, 1]
    })
    return SA_Trainer

def SA_WGP_Trainer_32_4_2a(params):
    params.update({
        'grids_g': [1, 4, 8, 16, 32],
        'grids_d': [32, 16, 8, 4, 1],
        'attn_spots' : [0, 1]
    })
    return SA_WGP_Trainer

def SA_WGP_Trainer_256_3_1a(params):
    params.update({
        'grids_g': [1, 4, 32, 256],
        'grids_d': [256, 32, 4, 1],
        'attn_spots': [0],
    })
    return SA_WGP_Trainer

def SA_Trainer_256_3_1a(params):
    params.update({
        'grids_g': [1, 4, 32, 256],
        'grids_d': [256, 32, 4, 1],
        'attn_spots': [0],
    })
    return SA_Trainer
