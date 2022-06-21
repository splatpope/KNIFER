import os
import math

import torch
from torchinfo import summary

def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def set_req_grads(model: torch.nn.Module, flag: bool):
    for params in model.parameters():
        params.requires_grad = flag

def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores, device=device),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

def is_pow2(n: int) -> bool:
    log2n = math.log2(n)
    return log2n == int(log2n)

def str_is_float(n:str) -> bool:
    try:
        float(n)
        return True
    except ValueError:
        return False

def model_summaries(depth=3, verbose=1):
    import knifer.context as KF
    DEVICE = "cuda" if torch.cuda.is_available() else 'cpu'
    N_CHANNELS = KF.ARCH.img_channels
    LATENT_SIZE = KF.ARCH.latent_size
    IMG_SIZE = KF.DATA.img_size
    BATCH_SIZE = KF.UPDATER.batch_size

    GEN_INPUT_SHAPE = (BATCH_SIZE, LATENT_SIZE, 1, 1)
    DISC_INPUT_SHAPE = (BATCH_SIZE, N_CHANNELS, IMG_SIZE, IMG_SIZE)
    
    print("\nGenerator :")
    summary(model=KF.ARCH.gen, 
        input_size=GEN_INPUT_SHAPE,
        depth=depth,
        device=DEVICE,
        verbose=verbose,
    )
    print("\nDiscriminator :")
    summary(model=KF.ARCH.disc,
        input_size=DISC_INPUT_SHAPE,
        depth=depth,
        device=DEVICE,
        verbose=verbose,
    )
