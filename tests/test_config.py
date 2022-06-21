from knifer import config as CFG

CONF_arch_dict = CFG.param_dict_from_file("configs/test_arch.yaml")
CONF_train_dict = CFG.param_dict_from_file("configs/test_updater.yaml")
CONF_arch = CFG.arch_param_from_dict(CONF_arch_dict)
CONF_train = CFG.training_param_from_dict(CONF_train_dict)
