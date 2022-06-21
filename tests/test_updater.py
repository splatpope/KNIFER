import test_arch # We need an architecture to have been created and loaded to context

from knifer.training.updater import GANUpdater
from test_config import CONF_train

import knifer.context as KF

UPD = GANUpdater(CONF_train)
KF.set_updater(UPD)
