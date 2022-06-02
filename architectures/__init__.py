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
