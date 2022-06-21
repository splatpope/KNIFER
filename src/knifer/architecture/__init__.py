from dataclasses import dataclass
import torch
import torch.nn as nn

from knifer.config import params as P
from . modules import ConvBlock

class GANModel(nn.Module):
    def __init__(self, params: P.ModelParameters):
        super().__init__()
        blocks = [ConvBlock(block_params) for block_params in params.blocks]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.blocks(input)

@dataclass
class GANArch():
    gen: GANModel
    disc: GANModel

    @property
    def latent_size(self) -> int:
        first_gen_block: ConvBlock = self.gen.blocks[0]
        return first_gen_block.conv.in_channels

    @property
    def img_channels(self) -> int:
        first_disc_block: ConvBlock = self.disc.blocks[0]
        return first_disc_block.conv.in_channels

    def serialize(self):
        return {
            "G_state": self.gen.state_dict(),
            "D_state": self.gen.state_dict(),
        }
   
def build(params: P.ArchParameters) -> GANArch:
    gen_model = GANModel(params.gen)
    disc_model = GANModel(params.disc)
    return GANArch(gen_model, disc_model)
