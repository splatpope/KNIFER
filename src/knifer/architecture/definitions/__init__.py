from dataclasses import dataclass
import inspect
from typing import Callable

import torch.nn as nn

import knifer.architecture.modules as kfm
import knifer.architecture.applicables as kfapps

NN_MODULES = {k.lower():v for k,v in vars(nn).items() if inspect.isclass(v)}
NN_ALIASES = {
    "bn2": "batchnorm2d",
    "in2": "instancenorm2d",
    "lrelu": "leakyrelu",
}
KF_MODULES = {k.lower():v for k,v in vars(kfm).items() if inspect.isclass(v)}
KF_ALIASES = {
    "attn": "attention",
}

# I LOVE OLD PYTHON I LOVE OLD PYTHON
#ALL_MODULES = NN_MODULES | KF_MODULES
NN_MODULES.update(KF_MODULES)
ALL_MODULES = NN_MODULES
#ALL_ALIASES = NN_ALIASES | KF_ALIASES
NN_ALIASES.update(KF_ALIASES)
ALL_ALIASES = NN_ALIASES

# TODO : this should be defines in arch.applicables
APPLICABLES = {
    "spectralnorm": kfapps.spectral_norm,
}
APPLICABLES_ALIASES = {
    "sn": "spectralnorm"
    #TODO : INITIALIZATION METHODS (see arch.applicables)
}

class MacroUndefinedError(ValueError):
    pass
class NoPreviousLayerError(IndexError):
    pass

#### Architecture definition #### 

@dataclass
class DefinitionMetadata():
    defines: dict
    kwarg_defaults: dict

@dataclass
class LayerDefinition():
    content: Callable
    def __call__(self) -> None:
        self.content()

@dataclass
class BlockDefinition():
    source: str
    content: "list[LayerDefinition]"
    def __repr__(self) -> str:
        return self.source

@dataclass
class ArchDefinition():
    gen_definition: "list[BlockDefinition]"
    disc_definition: "list[BlockDefinition]"
    metadata: DefinitionMetadata

class CompositeApplicable():
    def __init__(self, apps: "list[Callable]"):
        self.apps = apps
    
    def __call__(self, target: nn.Module) -> None:
        for app in self.apps:
            target.apply(app)
