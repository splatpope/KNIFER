import datetime
import torch
from torch.utils import data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from . import DCGANTrainer, WGAN_GPTrainer

## helper class to handle launching epochs, checkpointing, visualization
## meant to be used by the GUI
class TrainingManager():
    def __init__(self):
        self.epoch = 0
        self.dataset_folder = None

    def set_dataset_folder(self, folder):
        self.dataset_folder = folder

    def set_trainer(self, params):
        self.params = params
        arch = params["arch"]
        img_size = params["img_size"]
        ## TODO : allow for setting up transformations and augmentations
        self.dataset = dset.ImageFolder(self.dataset_folder, transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ]))
        if arch == "DCGAN":
            self.set_trainer_DCGAN(self.params)
        if arch == "WGAN_GP":
            self.set_trainer_WGAN_GP(self.params)
        self.fixed = self.trainer.get_fixed()

    ## TODO : rework training API to simply pass params and have the trainers check them
    ## this will greatly simplify this process : define a dict of all trainers, simply pass dataset+params to whichever matches arch
    def set_trainer_DCGAN(self, params):
        self.trainer = DCGANTrainer(
            self.dataset, 
            params["batch_size"],
            params["latent_size"],
            params["lr"],
            params["b1"],
            params["b2"],
        )

    def set_trainer_WGAN_GP(self, params):
        self.trainer = WGAN_GPTrainer(
            self.dataset, 
            params["batch_size"],
            params["latent_size"],
            params["lr"],
            params["b1"],
            params["b2"],
            params["critic_iters"],
            params["lambda_gp"],
        )

    def proceed(self, data, batch_id):
        try:
            (batch, labels) = next(data)
            self.batch = batch_id
            print("epoch", self.epoch, "batch", batch_id)
            self.trainer.process_batch(batch, labels)
            return batch_id + 1
        except StopIteration:
            self.epoch += 1
            return 0

    def synthetize_viz(self):
        fixed_fakes = self.trainer.GEN(self.fixed)
        #grid = vutils.make_grid(fixed_fakes, normalize=True)
        #grid_pil = transforms.ToPILImage()(grid).convert("RGB")
        vutils.save_image(fixed_fakes, normalize=True, fp=f"./viz/{self.get_filestamp()}.png")

    ## TODO : allow giving a path
    def save(self):
        state = {
            'dataset': self.dataset_folder,
            'model': self.trainer.serialize(),
            'params': self.params,
            'epoch': self.epoch,
            'batch': self.batch,
        }
        arch = state['params']['arch']
        epoch = state['epoch']
        batch = state['batch']
        torch.save(state, f"./savestates/{self.get_filestamp()}.pth")

    def load(self, path) -> int:
        state = torch.load(path)
        self.set_dataset_folder(state['dataset'])
        self.set_trainer(state['params'])
        self.trainer.deserialize(state['model'])
        self.epoch = state['epoch']
        self.batch = state['batch']
        return self.batch

    def get_filestamp(self) -> str:
        arch = self.params["arch"]
        epoch = self.epoch
        batch = self.batch
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        return f"{arch}_{timestamp}_{epoch}-{batch}"
