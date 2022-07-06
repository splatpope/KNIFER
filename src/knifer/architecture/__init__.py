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

class GANArch():
    def __init__(self, arch_def: ArchDefinition):
        self.source = arch_def
        self.gen = GANModel(arch_def.gen_definition)
        self.disc = GANModel(arch_def.disc_definition)

    @property
    def latent_size(self) -> int:
        return first_conv_in_model(self.gen).in_channels

    @property
    def img_channels(self) -> int:
        return first_conv_in_model(self.disc).in_channels

    def serialize(self) -> dict:
        return {
            "G_state": self.gen.state_dict(),
            "D_state": self.gen.state_dict(),
        }

    def __repr__(self) -> str:
        s = "Generator :\n"
        s += repr(self.gen) + "\n\n"
        s += "---------\n\n"
        s += "Discriminator :\n"
        s += repr(self.disc) + "\n"
        return s
