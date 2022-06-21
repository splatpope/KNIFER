import knifer.architecture as kfarch
import knifer.context as KF
from test_config import CONF_arch

ARCH = kfarch.build(CONF_arch)
KF.set_arch(ARCH)
