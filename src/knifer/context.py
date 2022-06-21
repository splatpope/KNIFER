import knifer.architecture as kfarch
import knifer.training.updater as kfupd
import knifer.training.data as kfdata
import knifer.io.logging as kflog

def set_arch(arch: kfarch.GANArch):
    global ARCH
    ARCH = arch

def set_logger(logger: kflog.GANLogger):
    global LOGGER
    LOGGER = logger

def set_updater(updater: kfupd.GANUpdater):
    if ARCH is None:
        raise ValueError
    global UPDATER
    UPDATER = updater

def set_data(data: kfdata.TrainingData):
    if UPDATER is None:
        raise ValueError
    global DATA
    DATA = data

def is_complete():
    components = [
        'ARCH',
        'UPDATER',
        'DATA',
        'LOGGER',
    ]
    return all([var in globals() for var in components])
