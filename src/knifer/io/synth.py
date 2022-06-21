import torch
import torchvision as tv

import knifer.context as KF

def denormalize(input: torch.Tensor):
    """ Denormalize [-1,1] generated images to [0,1]. """
    #return (input/2) + 0.5
    return tv.transforms.Normalize(-1, 2)(input)

def synth_n_fakes(n=1, z=None):
    """ Generate n fake images. You may provide your own latent vector."""
    if z is None:
        z = torch.randn(n, KF.ARCH.latent_size, 1, 1)
    with torch.no_grad():
        KF.ARCH.gen.eval()
        fakes = KF.ARCH.gen(z).to("cpu")
        return denormalize(fakes)
