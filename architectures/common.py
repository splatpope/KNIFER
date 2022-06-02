from email.mime import base
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#######################################################################################################

#### Misc utilities ####

def is_pow2(n):
    log2n = math.log2(n)
    return log2n == int(log2n)

#######################################################################################################

#### Trainer utilities ####

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

        pin_memory = torch.cuda.is_available()

        self.data = DataLoader(dataset, self.batch_size, 
            shuffle=True, 
            pin_memory=pin_memory, 
            num_workers=num_workers,
        )

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
            "upscales",
            "downscales",
            "features_g",
            "features_d",
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

class BaseGANModel(nn.Module):
    def __init__(self, params):
        super().__init__()
        check_required_params(self, params)

    def forward(self, input):
        raise NotImplementedError

    @classmethod
    def get_required_params(cls):
        return [
            "img_channels",
        ]

class BaseGenerator(BaseGANModel):
    @classmethod
    def get_required_params(cls):
        return super().get_required_params() + [
            "latent_size",
            "features_g",
            "upscales",
        ]

class BaseDiscriminator(BaseGANModel):
    @classmethod
    def get_required_params(cls):
        return super().get_required_params() + [
            "features_d",
            "downscales",
        ]

def check_required_params(o, params):
    for p in o.get_required_params():
        if p not in params:
            raise KeyError(p) ## should be caught by whatever tries to instanciate o

def disc_features(
        n_layers: int, 
        base_features: int, 
        features_list: "list[int]" = []
    ) -> "list[int]":
    """Derive list of features for generator tail layers.

    Args:
        n_layers (int): Amount of tail layers.
        base_features (int): Base amount of features. 
            (i.e. in_c of the first mid layer)
        feature_list (list[int], optional): List of in_c for the tail layers. 
            If missing, every layer will have double the preceding layer's features. 
            Defaults to None.
    """
    
    if not features_list:
        features_list = [base_features]
    assert features_list[0] == base_features, "Bogus features list."
    adds = [features_list[-1] * 2**(i+1) for i in range(n_layers - len(features_list))]
    return features_list + adds

def gen_features(
        n_layers: int, 
        base_features: int, 
        features_list: "list[int]" = []
    ) -> "list[int]":
    return disc_features(n_layers, base_features, features_list)[::-1]


class UpKConv2D(nn.ConvTranspose2d):
    def __init__(self, 
        in_c: int, 
        out_c: int, 
        K: int,
        bias: bool = True,
    ):
        """2D upscaling convolution layer of factor K (power of two)

        Args:
            in_c (int): Input channels.
            out_c (int): Output channels.
            K (int): Upscaling factor (power of two).
            bias (bool, optional): Uses bias. Defaults to True.
        """
        assert is_pow2(K), "Upscaling factor not power of 2"
        super().__init__(in_c, out_c,
            kernel_size = int(2*K),
            stride = int(K),
            padding = int(K/2),
            bias=bias
        )

class DownKConv2D(nn.Conv2d):
    def __init__(self, in_c: int, out_c: int, K: int, bias: bool = True):
        """2D downscaling convolution layer of factor K (power of two)

        Args:
            in_c (int): Input channels.
            out_c (int): Output channels.
            K (int): Downscaling factor (power of two).
            bias (bool, optional): Uses bias. Defaults to True.
        """
        assert is_pow2(K), "Downscaling factor not power of 2"
        super().__init__(in_c, out_c,
            kernel_size = int(2*K),
            stride = int(K),
            padding = int(K/2),
            bias=bias
        )

def doubling_arch_builder(img_size: int, base_features: int):
    """Builds required parameters for an architectures that does x2 upscales/downscales
    and doubles their layer feature, based on an image size and base number of features.

    Args:
        img_size (int): Image size for the models. (i.e. dataset and generator output)
        base_features (int): Base amount of conv layer features.

    Returns:
        Parameters (Tuple[int, int, int]): Upscale/Downscale factor list and features.
    """
    scalings = []
    size = 4
    while size < img_size:
        scalings.append(2)
        size *= 2
    features_d = disc_features(len(scalings), base_features)
    features_g = features_d[::-1]
    return scalings, features_d, features_g
    
