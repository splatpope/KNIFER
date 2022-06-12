import datetime
import glob
import gc
import os
from pathlib import Path
import contextlib

import torch
from torch.utils import data
import torchvision.datasets as dset
import torchvision.transforms as transforms

from torchinfo import summary

from architectures.common import doubling_arch_builder
from metrics import FID

from . trainer import GANTrainer
from . logging import GANLogger, NoTBError

#from . dataset import batch_mean_and_sd

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


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

# TODO : when there is support for first upscale/downscale =/= 4, check it here
def structure_check(p):
    if not p.keys() >= {"features_d", "features_g", "upscales", "downscales"}:
        s, fg, fd = doubling_arch_builder(p["img_size"], p["features"])
        p["features_g"] = fg
        p["features_d"] = fd
        p["upscales"] = s
        p["downscales"] = s
    else:
        from math import prod
        img_size = p["img_size"]
        assert prod(p["upscales"])*4 == img_size, "Upscales list doesn't produce image_size"
        assert prod(p["downscales"])*4 == img_size, "Downscales list doesn't reduce image_size"

# TODO : profiler
## helper class to handle launching epochs, checkpointing, visualization
class TrainingManager():
    def __init__(self, experiment, output, debug=False, parallel=False, use_tensorboard=True):
        self.output_path = output
        self.experiment_name = experiment ## TODO : integrate in save format along with arch, but this gonna kill back compat
        self.logger = GANLogger(experiment, output, use_tensorboard=use_tensorboard)

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
        
        # Premade datasets (e.g. from torchvision) are directly used
        if not premade:
            ## TODO : allow for setting up transformations and augmentations
            tr_combo = transforms.Compose([
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
            ])

            self.dataset = dset.ImageFolder(self.dataset_folder, transform=tr_combo)

            self.denorm = transforms.Normalize(
                mean= [-1, -1, -1],
                std= [2, 2, 2],
            )
        else:
            assert isinstance(premade, data.Dataset)
            self.dataset = premade

        # Trainer init 
        try:
            self.trainer = GANTrainer(self.dataset, params, num_workers=num_workers)
        # If any parameter is missing, the relevant error should rise here
        except KeyError as e:
            print(f"Parameter {e.args[0]} wrong or missing.")
        except Exception:
            raise

        if not self.trainer:
            print("Trainer initialization failed.")
            return False

        if self.parallel:
            self.trainer.parallelize()
        # We are assuming that all arch trainers use the GEN and DISC names
        self._log(self.trainer.GEN)
        self._log(self.trainer.DISC)

        # Get a random sample from the latent space in order to monitor qualitative progress
        self.fixed = self.trainer.get_fixed()

        # Write params to TB everytime we setup the trainer, I guess
        try:
            self.logger.write_params(params)
        except NoTBError as ntbe:
            print(ntbe)

        return True

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
    # not so simple anymore, i'm proud of you son
    # TODO : turn it into plain cli_train_loop
    def simple_train_loop(self, n_epochs=None, kimg=None):
        self.logger.init_stats() # to be safe
        # Make sure that models are in train mode
        self.trainer.GEN.train()
        self.trainer.DISC.train()
        # Handle epochs or kiloimages.
        # Actually not used, but hey
        if not n_epochs and not kimg:
            raise ValueError("Please enter either a number of epochs or a number of kiloimages to train for.")
        if n_epochs and kimg:
            raise ValueError("Do not set both number of epochs and kiloimages to train for.")
        if not n_epochs:
            n_epochs = (kimg*1000)//len(self.dataset) + 1 ## hacky

        dataloader = self.trainer.data

        start_epoch = self.epoch
        for _ in range(n_epochs):

            self.logger.init_stats() ## Reset loss averages

            self.epoch += 1
            self.logger.epoch = self.epoch
            self.batch = 0
            self._log(f"Epoch {self.epoch}/{start_epoch + n_epochs}")

            for batch, labels in tqdm(dataloader, dynamic_ncols=True):
                self.batch += 1
                self.logger.update_stats(*self.trainer.process_batch(batch, labels))

            self._log(self.logger.epoch_stats)

            try:
                self.logger.write_stats()
            except NoTBError as ntbe:
                print(ntbe)

        self.checkpoint = self.trainer.serialize()
        try:
            self.trainer.GEN.eval()
            self.synth_fixed(dest="tb")
        except NoTBError as ntbe:
            print(ntbe)

    def synth_fakes(self, n=1, z=None):
        if z is None:
            z = torch.randn(n, self.params["latent_size"], 1, 1)
        with torch.no_grad():
            fakes = self.trainer.GEN(z).cpu()
            return self.denorm(fakes)

    def synth_fixed(self, dest="storage"):
        fixed_fakes = self.synth_fakes(z=self.fixed)
        self.logger.save_samples(fixed_fakes, dest=dest)

    def save(self):
        if not self.checkpoint: # duh
            self._log("Nothing to save !")
            return False
        if not self.dataset_folder: # for testing purposes, ignore
            self._log("Not using a custom dataset, treating as dry run, not saving !")
        # Save state
        state = {
            'dataset': self.dataset_folder,
            'model': self.checkpoint,
            'params': self.params,
            'epoch': self.epoch,
            'kimg': self.kimg,
        }
        path = self.logger.save_checkpoint(state)
        self._log("Saved checkpoint at " + path)
        return True

    def load(self, path, premade=None) -> int:
        # Retrieve state
        if os.path.isdir(path): ## specify a directory to get the latest checkpoint inside
            files = glob.glob(path)
            path = max(files, key=os.path.getmtime)
        ## or just a file to load it directly
        state = self.logger.load_checkpoint(path)

        # Build manager from state
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

# this one is gonna be a little bit obsolete for now
    def get_filestamp(self) -> str:
        arch = self.params["arch"]
        epoch = self.epoch
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M')
        return f"{arch}_{timestamp}_{epoch}"

    def produce_FID(self, device='cpu'):
        from torch.utils.data import DataLoader

        BATCH_SIZE = 32
        DIMS = 2048
        DEVICE = device ## As GPU might very well be crowded

        # make a 1000 fakes dataloader
        fakes = self.synth_fakes(1000)
        fakes_dl = DataLoader(fakes, BATCH_SIZE, shuffle=True)

        # make a 1000 reals dataloader
        reals = DataLoader(self.dataset, 1000, shuffle=True)
        reals = next(iter(reals))[0]
        reals_dl = DataLoader(reals, BATCH_SIZE, shuffle=True)

        fid = FID(reals_dl, fakes_dl, BATCH_SIZE, DIMS, DEVICE)
        self._log(fid)
        try:
            self.logger.write_FID(fid, self.epoch)
        except NoTBError as ntbe:
            print(ntbe)
        
        return fid

    def model_summaries(self, depth=3, verbose=1):
        DEVICE = "cuda" if torch.cuda.is_available() else 'cpu'
        N_CHANNELS = self.trainer.channels
        LATENT_SIZE = self.trainer.latent_size
        IMG_SIZE = self.trainer.img_size
        BATCH_SIZE = self.trainer.batch_size

        GEN_INPUT_SHAPE = (BATCH_SIZE, LATENT_SIZE, 1, 1)
        DISC_INPUT_SHAPE = (BATCH_SIZE, N_CHANNELS, IMG_SIZE, IMG_SIZE)
        
        print("\nGenerator :")
        summary(model=self.trainer.GEN, 
            input_size=GEN_INPUT_SHAPE,
            depth=depth,
            device=DEVICE,
            verbose=verbose,
        )
        print("\nDiscriminator :")
        summary(model=self.trainer.DISC,
            input_size=DISC_INPUT_SHAPE,
            depth=depth,
            device=DEVICE,
            verbose=verbose,
        )
