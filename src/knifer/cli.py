import argparse
from knifer.architecture import build as build_arch
import knifer.context as KF
import knifer.config as CFG
import knifer.config.params as P
from knifer.training.updater import GANUpdater
from knifer.training.data import TrainingData
from knifer.io.logging import GANLogger
from knifer.experiment import Experiment

def init():
    parser = argparse.ArgumentParser(
        description="KNIFER Experimentation platform for GANs", 
    )
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--img-size", type=int)
    parser.add_argument("--arch-file", type=str)
    parser.add_argument("--training-file", type=str)
    parser.add_argument("--experiment", type=str)
    parser.add_argument("--storage-path", type=str)

    return parser.parse_args()

def create_context(args):
    arch_cfg = CFG.param_dict_from_file(args.arch_file)
    arch_params = CFG.arch_param_from_dict(arch_cfg)
    KF.set_arch(build_arch(arch_params))

    upd_cfg = CFG.param_dict_from_file(args.training_file)
    upd_params = CFG.training_param_from_dict(upd_cfg)
    KF.set_updater(GANUpdater(upd_params))

    data_params = CFG.P.DatasetParameters(args.dataset, args.img_size, None)
    KF.set_data(TrainingData(data_params))

def create_experiment(args):
    create_context(args)
    exp_params = P.ExperimentParameters(args.experiment, args.storage_path)

    KF.set_logger(GANLogger(args.experiment, args.storage_path))
    return Experiment(exp_params)