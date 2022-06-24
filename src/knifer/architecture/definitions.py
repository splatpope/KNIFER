import re
import inspect
import typing
import torch.nn as nn
import torch.nn.utils.parametrizations as nnP

from . import modules as kfm
from . import applicables as kfapps
from knifer.misc_utils import str_is_float

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

Arg2DInt = typing.Tuple[int, int]
ArgIntOr2DInt = typing.Union[int,Arg2DInt]

class MacroUndefinedError(ValueError):
    pass
class ApplicableError(Exception):
    pass
class NoPreviousModuleError(IndexError):
    pass
class WrongTypeError(TypeError):
    pass

def substitute_macros(expression:str, defines: dict) -> str:
    for k, v in defines.items():
        expression = expression.replace(f'${k}', str(v))
    if '$' in expression:
        raise MacroUndefinedError
    return expression

def handle_applicables(fn_names: "list[str]", target: nn.Module):
    fn_names = [APPLICABLES_ALIASES.get(n, n) for n in fn_names]
    functions = [APPLICABLES[n] for n in fn_names]

    for f in functions:
        target.apply(f)

def decode_block_definition(
        block_def: str, 
        defines: dict = None, 
        kwarg_defaults: dict=None,
    ) -> nn.Sequential:
    layer_defs = block_def.strip().split('\n')
    layer_defs = [substitute_macros(ld, defines) for ld in layer_defs]

    block = nn.Sequential()
    
    # TODO : split this in a decode_layer_definition function maybe ?
    for ld in layer_defs:
        tokens = ld.split()
        name, arg_str_list = tokens[0], tokens[1::]

        # Retrieve applicables and apply them
        if name.lower() == "apply":
            try:
                previous_layer = block[-1]
                handle_applicables(arg_str_list, previous_layer)
            except IndexError:
                raise NoPreviousModuleError
            finally:
                continue

        args = kwarg_defaults.get(name, dict())

        name = ALL_ALIASES.get(name, name)
        module = ALL_MODULES[name.lower()]

        # Construct an argument dict for the current layer
        # using its module's init signature
        sig = inspect.signature(module)
        params = [(t.name, t.annotation) for t in sig.parameters.values()]

        for i, arg in enumerate(arg_str_list):
            try:
                arg_name = params[i][0]
                arg_t = params[i][1]
                # Handle special type annotations
                if arg_t == ArgIntOr2DInt:
                    arg_t = int
                if arg_t == inspect._empty:
                    arg_t = int # might be a bad idea
                args.update({arg_name:arg_t(arg)})
            except TypeError:
                raise WrongTypeError

        block.append(module(**args))
    
    return block

