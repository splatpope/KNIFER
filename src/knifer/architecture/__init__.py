from dataclasses import dataclass
import torch
import torch.nn as nn

from . definitions import *

class GANModel(nn.Module):
    def __init__(self, block_defs: "list[BlockDefinition]"):
        super().__init__()
        blocks = list()
        for bid, block_def in enumerate(block_defs):
            layers = list()
            for lid, layer_def in enumerate(block_def.content):
                if isinstance(layer_def.content, CompositeApplicable):
                    if not layers:
                        raise NoPreviousLayerError
                    layer_def.content(layers[-1])
                else:
                    layers.append(layer_def.content())
            blocks.append(nn.Sequential(*layers))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.blocks(input)

def conv_in_sequential(target: nn.Sequential):
    for m in target:
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            return m

def first_conv_in_model(target: GANModel):
    return conv_in_sequential(target.blocks[0])

@dataclass
class GANArch():
    gen: GANModel
    disc: GANModel

    @property
    def latent_size(self) -> int:
        return first_conv_in_model(self.gen).in_channels

    @property
    def img_channels(self) -> int:
        return first_conv_in_model(self.disc).in_channels

    def serialize(self):
        return {
            "G_state": self.gen.state_dict(),
            "D_state": self.gen.state_dict(),
        }
   
def build(arch_def: ArchDefinition) -> GANArch:
    gen_model = GANModel(arch_def.gen_definition)
    disc_model = GANModel(arch_def.disc_definition)
    return GANArch(gen_model, disc_model)
