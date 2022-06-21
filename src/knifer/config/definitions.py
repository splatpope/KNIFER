import re
import inspect
import torch.nn as nn

from knifer.misc_utils import str_is_float

nn_classes = {k.lower():v for k,v in vars(nn).items() if inspect.isclass(v)}
nn_aliases = {
    "bn2": "batchnorm2d",
    "in2": "instancenorm2d",
    "lrelu": "leakyrelu",
}

def conv_layer_definition(nature: str, factor: int):
    match nature:
        case "decode":
            return {
                "conv_module": nn.ConvTranspose2d,
                "kernel_size": factor,
                "stride": 1,
                "padding": 0,
            }
        case "score":
            return {
                "conv_module": nn.Conv2d,
                "kernel_size": factor,
                "stride": 1,
                "padding": 0,
            }
        case "up":
            return {
                "conv_module": nn.ConvTranspose2d,
                "kernel_size": 2 * factor,
                "stride": factor,
                "padding": factor // 2,
            }
        case "down":
            return {
                "conv_module": nn.Conv2d,
                "kernel_size": 2 * factor,
                "stride": factor,
                "padding": factor // 2,
            }

NORM_LAYERS = {
    "bnorm" : nn.BatchNorm2d,
    "inorm": nn.InstanceNorm2d,
}

ACT_FNS = {
    "relu": nn.ReLU,
    "lrelu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}

def decode_block_definition(block_def: str) -> dict:
    """ Create a dictionnary for a convolution block's parameters. """
    # Decode the first line of the block, describing the convolution layer
    conv_def = block_def.pop(0)
    conv_bias = "bnorm" in block_def

    conv_tokens = conv_def.split()

    conv_nature = conv_tokens.pop(0)
    conv_nature = re.split(r'(\d+)', conv_nature)
    conv_nature, conv_factor = conv_nature[0], int(conv_nature[1])

    conv_params = conv_layer_definition(conv_nature, conv_factor)

    conv_params.update({
        "in_channels": int(conv_tokens.pop(0)),
        "out_channels": int(conv_tokens.pop(0)),
    })

    norm_layer = None
    activation = None
    spectral_norm = False
    self_attention = False

    while conv_tokens:
        option = conv_tokens.pop(0)
        match option:
            case "sn":
                spectral_norm = True
    
    while block_def:
        layer = block_def.pop(0)
        if layer in NORM_LAYERS:
            norm_layer = NORM_LAYERS[layer]
        else:
            layer_tokens = layer.split()
            layer_nature = layer_tokens.pop(0)
            if layer_nature in ACT_FNS: # Handle activation functions
                act_fn_module = ACT_FNS[layer_nature]
                act_fn_options = dict()
                while layer_tokens:
                    option = layer_tokens.pop(0)
                    if str_is_float(option) and layer_nature == "lrelu":
                        act_fn_options.update({"negative_slope": float(option)})
                    elif option == "inplace":
                        act_fn_options.update({"inplace": True})
                activation = act_fn_module(**act_fn_options)
            elif layer_nature == "attn":
                self_attention = True

    conv_params.update({
        "norm_layer": norm_layer,
        "activation": activation,
        "spectral_norm": spectral_norm,
        "self_attention": self_attention,
    })

    return conv_params
