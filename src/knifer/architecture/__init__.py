from dataclasses import dataclass
import torch
import torch.nn as nn

from knifer.config import params as P

class GANModel(nn.Module):
    def __init__(self, params: P.ModelParameters):
        super().__init__()
        if isinstance(params.blocks, nn.ModuleList):
            self.blocks = nn.Sequential(*params.blocks)
        else:
            self.blocks = params.blocks

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
   
def build(params: P.ArchParameters) -> GANArch:
    gen_model = GANModel(params.gen)
    disc_model = GANModel(params.disc)
    return GANArch(gen_model, disc_model)
