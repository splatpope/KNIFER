from dataclasses import dataclass
import yaml

from . definitions import decode_block_definition
from . import params as P

def training_param_from_dict(trainer_cfg: dict) -> P.UpdaterParameters:
    #### Trainer ####
    opt_g = trainer_cfg["opt_g"]
    opt_d = trainer_cfg["opt_d"]

    opt_type2class = {
        "adam": P.AdamParameters,
    }
    opt_g_type = opt_type2class.get(opt_g["type"], P.OptimParameters)
    opt_d_type = opt_type2class.get(opt_g["type"], P.OptimParameters)

    regularization = trainer_cfg.get("regularization", None)
    
    tp_dict = {
        "batch_size": trainer_cfg["batch_size"],
        "opt_G": opt_g_type(**opt_g["params"]),
        "opt_D": opt_d_type(**opt_d["params"]),
        "loss": trainer_cfg["loss"],
        "regularization": regularization,
    }

    return P.UpdaterParameters(**tp_dict)


def arch_param_from_dict(arch_cfg: dict) -> P.ArchParameters:
    """ Create architecture parameters from a dictionnary. """
    gen_def = arch_cfg["gen"]["layers"]
    disc_def = arch_cfg["disc"]["layers"]

    gen_blocks = [decode_block_definition(block_def) for block_def in gen_def]
    disc_blocks = [decode_block_definition(block_def) for block_def in disc_def]

    gen_block_params = [P.ConvBlockParameters(**block) for block in gen_blocks]
    disc_block_params = [P.ConvBlockParameters(**block) for block in disc_blocks]

    arch_params = {
        "latent_size": gen_block_params[0].in_channels,
        "gen": P.ModelParameters(gen_block_params),
        "disc": P.ModelParameters(disc_block_params)
    }

    return P.ArchParameters(**arch_params)


def param_dict_from_file(cfg_path: str) -> dict:
    """ Create architecture parameters from a YAML config file. """
    with open(cfg_path, 'r') as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e) # TODO : replace with logging
