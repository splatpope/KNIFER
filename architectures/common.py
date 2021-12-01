import datetime
import torch
from torch.utils import data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from . import DCGANTrainer

## helper class to handle launching epochs, checkpointing, visualization
## meant to be used by the GUI
class TrainingManager():
    def __init__(self):
        self.epoch = 0
        self.dataset_folder = None

    def set_dataset_folder(self, folder):
        self.dataset_folder = folder

    def set_trainer(self, arch, img_size, batch_size, latent_size, params):
        self.arch = arch
        self.image_size = img_size
        self.dataset = dset.ImageFolder(self.dataset_folder, transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ]))
        self.batch_size = batch_size
        self.latent_size = latent_size
        if self.arch == "DCGAN":
            self.set_trainer_DCGAN(params)

    def set_trainer_DCGAN(self, params):
        self.trainer = DCGANTrainer(
            self.dataset, 
            self.batch_size,
            self.latent_size,
            params["lr"],
            params["b1"],
            params["b2"],
        )
        self.fixed = self.trainer.get_fixed()

    def proceed(self, data, batch_id):
        try:
            (batch, labels) = next(data)
            print("epoch", self.epoch, "batch", batch_id)
            self.trainer.process_batch(batch, labels)
            return batch_id + 1
        except StopIteration:
            self.epoch += 1
            return 0

    def save(self):
        state = {
            'epoch': self.epoch,
            'model': self.trainer.serialize(),
        }
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

        torch.save(state, f"./savestates/{self.arch}_{timestamp}_{self.epoch}.pth")

    def load(self, path):
        state = torch.load(path)
        self.epoch = state['epoch']
        self.trainer.deserialize(state['model'])

