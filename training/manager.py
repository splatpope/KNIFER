import datetime
import glob
import os
from pathlib import Path

import torch
from torch.utils import data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

from architectures import DCGANTrainer, SAGANTrainer_256_3_1a_WGP, SAGANTrainer_32_4_2a, SAGANTrainer_32_4_2a_WGP, WGAN_GPTrainer
from architectures import DCGANTrainerEZMnist
from architectures import DCGANTrainer_256_3, WGAN_GPTrainer_256_3

#TODO : allow for setting number of features for models
# currently equal to biggest grid size
#TODO : save current batch id AND ordered data; OR simply allow saving only on epoch change
# this is to prevent messing with the ordering of mid-epoch data used for training
# which defeats the purpose of stochastic batching

KNIFER_ARCHS = {
    "DCGAN": DCGANTrainer,
    "DCGAN_256_3": DCGANTrainer_256_3,
    "DCGAN_EZMNIST": DCGANTrainerEZMnist,
    "WGAN_GP": WGAN_GPTrainer,
    "WGAN_GP_256_3": WGAN_GPTrainer_256_3,
    "SAGAN_TEST": SAGANTrainer_32_4_2a,
    "SAGAN_TEST_WGP": SAGANTrainer_32_4_2a_WGP,
    "SAGAN_TEST_WGP_256_3": SAGANTrainer_256_3_1a_WGP,
}

## helper class to handle launching epochs, checkpointing, visualization
## meant to be used by the GUI, but can be used alone
class TrainingManager():
    def __init__(self, debug=False):
        self.epoch = 0
        self.checkpoint = None
        self.dataset_folder = None
        self.debug = debug

    def _log(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    def set_dataset_folder(self, folder):
        self.dataset_folder = folder

    def set_trainer(self, params, premade = None):
        self.trainer = None
        if (torch.cuda.is_available()): ## may or may not work
            torch.cuda.empty_cache()
        self.epoch = 0
        self.kimg = 0
        self.batch = 0
        self.params = params
        arch = params["arch"]
        img_size = params["img_size"]
        ## TODO : allow for setting up transformations and augmentations
        tr_combo = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ])
        if not premade:
            self.dataset = dset.ImageFolder(self.dataset_folder, transform=tr_combo)
        else:
            assert isinstance(premade, data.Dataset)
            self.dataset = premade

        try:
            arch_to_go = KNIFER_ARCHS[arch]
        except KeyError as e:
            print(f"Architecture {e.args[0]} not found. Please provide an architecture present in KNIFER_ARCHS.")

        try:
            self.trainer = arch_to_go(self.dataset, params) 
            ## may lead to unused params being passed
            ## who cares ?
            self.trainer.build(params)
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
        if not n_epochs and not kimg:
            raise ValueError("Please enter either a number of epochs or a number of kiloimages to train for.")
        if n_epochs and kimg:
            raise ValueError("Do not set both number of epochs and kiloimages to train for.")

        dataloader = self.trainer.data

        if not n_epochs:
            n_epochs = (kimg*1000)//len(self.dataset) + 1 ## hacky

        start_epoch = self.epoch
        for _ in range(n_epochs):
            self.epoch += 1
            self.batch = 0
            self._log(f"Epoch {self.epoch}/{start_epoch + n_epochs}")
            for batch, labels in tqdm(dataloader):
                self.batch += 1
                self.trainer.process_batch(batch, labels)
            self.checkpoint = self.trainer.serialize()

    def synthetize_viz(self, dest=None):
        if not dest:
            dest = "./viz/"
        dest = Path(dest)
        filename = self.get_filestamp() + ".png"
        path = dest / filename
        with torch.no_grad():
            self.trainer.GEN.eval()
            fixed_fakes = self.trainer.GEN(self.fixed)
            #grid = vutils.make_grid(fixed_fakes, normalize=True)
            #grid_pil = transforms.ToPILImage()(grid).convert("RGB")
            vutils.save_image(fixed_fakes, fp=path)

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
