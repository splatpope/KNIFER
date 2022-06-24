from knifer.architecture.definitions import decode_block_definition
from knifer.config import param_dict_from_file

contents = param_dict_from_file("./test_arch_custom.yaml")
defines = contents["defines"]
kwarg_defaults = contents["kwarg_defaults"]

gen_block_defs = contents["gen"]
disc_block_defs = contents["disc"]

gen_blocks = [decode_block_definition(bd, defines, kwarg_defaults) for bd in gen_block_defs]
disc_blocks = [decode_block_definition(bd, defines, kwarg_defaults) for bd in disc_block_defs]
