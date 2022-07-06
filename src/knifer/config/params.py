import sys
from dataclasses import dataclass

import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms

#### Model update process definition ####
## TODO : take care of all this like with models
## because they aren't really parameters, they're factories
@dataclass
class OptimParameters():
    lr: float

    def __call__(self, model:nn.Module):
        return optim.SGD(model.parameters(), lr = self.lr)

@dataclass
class AdamParameters(OptimParameters):
    betas: "list[float]"
    weight_decay: float=0.0

    def __call__(self, model:nn.Module):
        return optim.Adam(
                model.parameters(), 
                lr = self.lr, 
                betas = tuple(self.betas), 
                weight_decay = self.weight_decay,
            )

@dataclass 
class UpdaterParameters():
    batch_size: int
    opt_G: OptimParameters
    opt_D: OptimParameters
    loss: str
    regularization: str

#### Dataset parameters ####

@dataclass
class AugmentationParameters():
    transforms: "list[nn.Module]"

@dataclass
class DatasetParameters():
    path: str
    img_size: int
    aug: AugmentationParameters

#### Experiment parameters ####

@dataclass
class ExperimentParameters():
    name: str
    storage_path: str
    
@dataclass
class TrainingParameters():
    n_epochs: int
    synth_steps: int = sys.maxsize
    save_steps: int = sys.maxsize
    metrics_steps: int = sys.maxsize
    metrics: "list[str]" = None

## TODO : make it so that params are saved in the context for information purposes
## TODO : learning rate scheduling...
