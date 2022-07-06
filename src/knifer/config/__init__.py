import yaml

from knifer.architecture.definitions import ArchDefinition, DefinitionMetadata
from knifer.architecture.definitions.codec import decode_arch_definition
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


def arch_def_from_dict(arch_cfg: dict) -> ArchDefinition:
    """ 
    Create an architecture definition from a dictionary of string representations.
    """
    gen_def_raw = arch_cfg["gen"]
    disc_def_raw = arch_cfg["disc"]
    md = DefinitionMetadata(arch_cfg["defines"], arch_cfg["kwarg_defaults"])

    return decode_arch_definition(gen_def_raw, disc_def_raw, md)


def param_dict_from_file(cfg_path: str) -> dict:
    """ Retrieve a parameter dictionary from a YAML config file. """
    import knifer.context as KF
    with open(cfg_path, 'r') as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            KF.LOGGER().exception(e)
