import ast
import re
from functools import partial

from . import *


def compose_applicables(fn_names: "list[str]") -> CompositeApplicable:
    fn_names = [APPLICABLES_ALIASES.get(n, n) for n in fn_names]
    functions = [APPLICABLES[n] for n in fn_names]

    return CompositeApplicable(functions)

def substitute_macros(expression:str, defines: dict) -> str:
    pattern = re.compile(r"\s\$(\w+)\b")
    macros = pattern.findall(expression)
    for macro in macros:
        if macro not in defines:
            raise MacroUndefinedError(macro)
        value = defines[macro]
        expression = re.sub(rf"\${macro}(\b)", rf"{value}\1", expression)
    return expression

def decode_layer_definition(raw_layer_def: str, metadata: DefinitionMetadata):
    """ 
    Generate a factory for the layer defined by the input string, which contains
    the layer's module or special function first, then its arguments in order.
    Metadata contains macros (for better structuring of the definition file) and
    default arguments for some kinds of layers (typically ones that are constant
    among the same layer types).
    """

    # First things first, expand all macros with their values.
    layer_def_expanded = substitute_macros(raw_layer_def, metadata.defines)
    # Extract the tokens : the layer's "name" and arguments.
    tokens = layer_def_expanded.split()
    func, arg_list_str = tokens[0], tokens[1::]
    # Special layer name "apply" : we want to apply a function or many functions to
    # the previous layer.
    if func.lower() == "apply":
        return LayerDefinition(layer_def_expanded, compose_applicables(arg_list_str))
    
    # Retrieve the default args if there are any for this layer type.
    # This will serve as a base for our final arguments dict that is going
    # to be used in the factory.
    kwargs = metadata.kwarg_defaults.get(func, dict())
    # Retrieve the layer's corrresponding module, through an eventual alias.
    func = ALL_ALIASES.get(func, func)
    module = ALL_MODULES[func.lower()]
    # Retrieve the parameters for constructing the layer.
    sig = inspect.signature(module)
    params = [(t.name, t.annotation) for t in sig.parameters.values()]

    # At this point, we know which arguments we need to provide in which order.
    # Now we need to insantiate them from their string representation.
    for idx, arg_str in enumerate(arg_list_str):
        arg_name = params[idx][0]
        #arg_type = params[idx][1]

        arg = ast.literal_eval(arg_str)
        kwargs.update({arg_name:arg})


    return LayerDefinition(layer_def_expanded, partial(module, **kwargs))


def decode_block_definition(raw_block_def: str, metadata: DefinitionMetadata) -> list:
    raw_layer_defs = raw_block_def.strip().split('\n')
    layer_defs = [decode_layer_definition(rld, metadata) for rld in raw_layer_defs]

    return BlockDefinition(raw_block_def, layer_defs)

def decode_arch_definition(
    raw_gen: "list[str]", 
    raw_disc: "list[str]", 
    metadata: DefinitionMetadata,
    ):

    gen_def = [decode_block_definition(rgbd, metadata) for rgbd in raw_gen]
    disc_def = [decode_block_definition(rdbd, metadata) for rdbd in raw_disc]

    return ArchDefinition(gen_def, disc_def, metadata)


# TODO: encoders
