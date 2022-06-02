import datetime
import glob
import gc
import os
from pathlib import Path
from platform import architecture

import torch
from torch.utils import data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from architectures.common import BaseTrainer, doubling_arch_builder

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

# ensure that lr_g and lr_d are set if there's either of them set
# or learning_rate is set (it is then used for both)
def lr_check(p):
    if "lr_g" not in p and "lr_d" not in p:
        if "learning_rate" in p:
            p["lr_g"] = p["learning_rate"]
            p["lr_d"] = p["learning_rate"]
    if "lr_g" not in p and "lr_d" in p:
        p["lr_g"] = p["lr_d"]
    if "lr_d" not in p and "lr_g" in p:
        p["lr_d"] = p["lr_g"]

# TODO : actually make sure that the inputs are coherent
# when not generating them
def structure_check(p):
    if not p.keys() >= {"features_d", "features_g", "upscales", "downscales"}:
        s, fg, fd = doubling_arch_builder(p["img_size"], p["features"])
        p["features_g"] = fg
        p["features_d"] = fd
        p["upscales"] = s
        p["downscales"] = s


from architectures import *

KNIFER_ARCHS = {
    "DCGAN": DC_Trainer_default,
    "WGAN_GP": WGP_Trainer_default,
    "SAGAN": SA_Trainer_default,
    "SAGAN_WGP": SA_WGP_Trainer_default,
}

## helper class to handle launching epochs, checkpointing, visualization
class TrainingManager():
    def __init__(self, experiment, debug=False, parallel=False):
        self.experiment_name = experiment ## TODO : integrate in save format along with arch, but this gonna kill back compat
        self.epoch = 0
        self.checkpoint = None
        self.dataset_folder = None
        self.debug = debug

        self.parallel = parallel

    def _log(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    def set_dataset_folder(self, folder):
        self.dataset_folder = folder

    def set_trainer(self, params, premade = None, num_workers=0):
        self.trainer = None
        if (torch.cuda.is_available()): ## may or may not work
            gc.collect()
            torch.cuda.empty_cache()
        self.epoch = 0
        self.kimg = 0
        self.batch = 0
        self.params = params
        arch = params["arch"]
        img_size = params["img_size"]
        
        lr_check(params)
        structure_check(params)

        ## TODO : allow for setting up transformations and augmentations
        tr_combo = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ])

        # Premade datasets (e.g. from torchvision) are directly used
        if not premade:
            self.dataset = dset.ImageFolder(self.dataset_folder, transform=tr_combo)
        else:
            assert isinstance(premade, data.Dataset)
            self.dataset = premade

        # Is the architecture provided actually present ?
        try:
            arch_to_go: BaseTrainer = KNIFER_ARCHS[arch](params) ## Get the chosen architecture class, mangling parameters if needed
        except KeyError as e:
            print(f"Architecture {e.args[0]} not found. Please provide an architecture present in KNIFER_ARCHS.")
        
        # Trainer init 
        try:
            self.trainer = arch_to_go(self.dataset, params, num_workers)
        # If any parameter is missing, the relevant error should rise here
        except KeyError as e:
            print(f"Parameter {e.args[0]} required by {arch}.")
        except Exception:
            ## something else happened, figure it out bro
            raise

        if not self.trainer:
            print("Trainer initialization failed.")
            return

        ## We are assuming that all arch trainers use the GEN and DISC names
        self._log(self.trainer.GEN)
        self._log(self.trainer.DISC)
        ## Get a random sample from the latent space in order to monitor qualitative progress
        self.fixed = self.trainer.get_fixed()

    # this function is only for usage with the GUI, which both need to be interruptible
    def proceed(self, data, batch_id):
        try:
            (batch, labels) = next(data)
            self.batch = batch_id
            self._log("epoch", self.epoch, "batch", batch_id)
            self.trainer.process_batch(batch, labels)
            return batch_id + 1
        except StopIteration:
            self.checkpoint = self.trainer.serialize()
            self.epoch += 1
            return 0

    # this function is more adapted to command line training
    def simple_train_loop(self, n_epochs=None, kimg=None):
        self.trainer.GEN.train()
        self.trainer.DISC.train()
        ## Handle epochs or kiloimages. 
        if not n_epochs and not kimg:
            raise ValueError("Please enter either a number of epochs or a number of kiloimages to train for.")
        if n_epochs and kimg:
            raise ValueError("Do not set both number of epochs and kiloimages to train for.")
        if not n_epochs:
            n_epochs = (kimg*1000)//len(self.dataset) + 1 ## hacky

        dataloader = self.trainer.data

        start_epoch = self.epoch
        for _ in range(n_epochs):
            self.epoch += 1
            self.batch = 0
            self._log(f"Epoch {self.epoch}/{start_epoch + n_epochs}")
            for batch, labels in tqdm(dataloader, dynamic_ncols=True):
                self.batch += 1
                self.trainer.process_batch(batch, labels)
            self.checkpoint = self.trainer.serialize()

    def synthetize_viz(self, dest=None):
        if not dest:
            dest = "./viz/"
            if self.experiment_name:
                dest += self.experiment_name
        dest = Path(dest)
        filename = self.get_filestamp() + ".png"
        path = dest / filename
        with torch.no_grad():
            self.trainer.GEN.eval()
            fixed_fakes = self.trainer.GEN(self.fixed)
            #grid = vutils.make_grid(fixed_fakes, normalize=True)
            #grid_pil = transforms.ToPILImage()(grid).convert("RGB")
            make_folder(dest)
            vutils.save_image(fixed_fakes.cpu(), fp=path)

    def save(self, dest=None):
        if not self.checkpoint:
            self._log("Nothing to save !")
            return False
        if not self.dataset_folder:
            self._log("Not using a custom dataset, treating as dry run, not saving !")
        state = {
            'dataset': self.dataset_folder,
            'model': self.checkpoint,
            'params': self.params,
            'epoch': self.epoch,
            'kimg': self.kimg,
        }
        if not dest:
            dest = "./savestates/"
        dest = Path(dest)
        filename = self.get_filestamp() + ".pth"
        path = dest / filename
        make_folder(dest)
        torch.save(state, path)
        return path

    def load(self, path, premade=None) -> int:
        if os.path.isdir(path): ## specify a directory to get the latest checkpoint inside
            files = glob.glob(path)
            path = max(files, key=os.path.getmtime)
        ## or just a file to load it directly
        state = torch.load(path)
        self.set_dataset_folder(state['dataset'])
        if not self.dataset_folder and not premade:
            self._log("ERROR: checkpoint needs a premade dataset to be specified!")
            return False
        else:
            self.set_trainer(state['params'], premade)
        self.trainer.deserialize(state['model'])
        self.epoch = state['epoch']
        self.kimg = state['kimg']
        self.batch = 0
        return self.batch

    def get_filestamp(self) -> str:
        arch = self.params["arch"]
        epoch = self.epoch
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M')
        return f"{arch}_{timestamp}_{epoch}"
