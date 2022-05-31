import math
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm
import torch.optim as optim
from torch.utils.data import DataLoader


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BaseTrainer():
    def __init__(self, dataset, params: dict, num_workers=0):
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

        sample = dataset[0][0]

        self.img_size = sample.shape[2]
        if not "img_size" in params: ## should never happen but hey
            params["img_size"] = self.img_size

        self.channels = sample.shape[0]
        if not "img_channels" in params:
            params["img_channels"] = self.channels

        grids_from_params_or_default(params)

        pin_memory = torch.cuda.is_available()

        self.data = DataLoader(dataset, self.batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers)

        self.build(params)

    def build(self, params):
        pass

    def process_batch(self, value, labels):
        pass

    @classmethod
    def get_required_params(cls):
        return [
            "batch_size",
            "latent_size",
            "lr_g",
            "lr_d",
            "b1",
            "b2",
        ]
    
    def get_fixed(self):
        return torch.randn(self.batch_size, self.latent_size, 1, 1, device=DEVICE)

    def parallelize(self): # don't use for now
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

#######################################################################################################

#### model utilities ####
def check_required_params(o, params):
    for p in o.get_required_params():
        if p not in params:
            raise KeyError(p) ## should be caught by whatever tries to instanciate o

def grids_from_params_or_default(params: dict):
        """Ensure that params contain layer size transitions for G and D. 
        Generate meaningful defaults otherwise.

        Args:
            params (dict): model hyperparameters

        Raises:
            AttributeError: img_size needs to be known (or the params are malformed anyway)

        Returns:
            list, list : the layer size transitions
        """
        if "img_size" not in params:
            raise AttributeError
        img_size = params["img_size"]

        if "grids_g" not in params:
            grids_g = [1]
            i = 4
            while i <= img_size:
                grids_g.append(i)
                i *= 2
            params["grids_g"] = grids_g
        else:
            validate_grids(params["grids_g"])

        assert(params["grids_g"][-1] == img_size) # no weird stuff happened while building grids_g ?

        if "grids_d" not in params:
            grids_d = grids_g[::-1]
            params["grids_d"] = grids_d
        else:
            validate_grids(params["grids_d"])

        assert(params["grids_d"][0] == img_size) # ditto

## A list of integer grid sizes is valid if they are all powers of 2 and are monotonically arranged
def validate_grids(grids):
    assert(isinstance(grids, list))
    assert(all([isinstance(g, int) for g in grids]))
    if any([not math.log2(g).is_integer() for g in grids]):
        return False
    if grids == sorted(grids) or grids == sorted(grids, reverse=True):
        return True
    return False

## Compute grid transitions to get layer parameters
def layers(grids):
    transitions = []
    if grids[0] < grids[1]: # upscaling
        for i in range(len(grids)-1):
            transitions.append(int(grids[i+1]/grids[i]))
    else: # downscaling
        for i in range(len(grids)-1):
            transitions.append(int(grids[i]/grids[i+1]))
    return transitions

def ReSample(in_c, out_c, factor, bias=True, dir="up", spectral = False):
    assert(factor >= 2)
    assert(math.log2(factor).is_integer())
    k = int(2*factor)
    s = int(factor)
    p = int(factor/2)
    if dir == "up":
        op = nn.ConvTranspose2d(
            in_channels=in_c, 
            out_channels=out_c, 
            kernel_size=k, 
            stride=s, 
            padding=p, 
            bias=bias
        )
    elif dir == "down":
        op = nn.Conv2d(
            in_channels=in_c, 
            out_channels=out_c, 
            kernel_size=k, 
            stride=s, 
            padding=p, 
            bias=bias
        )
    if spectral:
        op = spectral_norm(op)
    return op

def UpSample(in_c, out_c, factor, bias=True, spectral=False):
    return ReSample(in_c, out_c, factor, bias, dir="up", spectral=spectral)

def DownSample(in_c, out_c, factor, bias=True, spectral=False):
    return ReSample(in_c, out_c, factor, bias, dir="down", spectral=spectral)
