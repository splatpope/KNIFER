
from pathlib import Path

from torch import save as chkpt_save, load as chkpt_load
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from misc_utils import make_folder

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class GANEpochLossMeter:
    def __init__(self):
        self.loss_G = AverageMeter()
        self.loss_D = AverageMeter() # may include GP
        self.loss_D_real = AverageMeter()
        self.loss_D_fake = AverageMeter()

    def update(self, lg, ld, ldr, ldf):
        self.loss_G.update(lg)
        self.loss_D.update(ld)
        self.loss_D_real.update(ldr)
        self.loss_D_fake.update(ldf)

    def write(self, writer: SummaryWriter, epoch: int):
        writer.add_scalar('Generator_Loss', self.loss_G.avg, epoch)
        writer.add_scalar('Discriminator_Loss_Total', self.loss_D.avg, epoch)
        writer.add_scalar('Discriminator_Loss_Real', self.loss_D_real.avg, epoch)
        writer.add_scalar('Discriminator_Loss_Fake', self.loss_D_fake.avg, epoch)

    def __str__(self):
        return f"Loss G: {self.loss_G.avg} | Loss D (Total/Real/Fake) : {self.loss_D.avg} / {self.loss_D_real.avg} / {self.loss_D_fake.avg}"

class NoTBError(AttributeError):
    def __init__(self, *args):
        super().__init__("Trying to use tensorboard writer, but it was explicitly disabled.")

class GANLogger():
    def __init__(self, experiment, use_tensorboard=False):
        self.epoch = 0

        self.experiment = experiment
        self.exp_path = Path("experiments") / experiment
        make_folder(self.exp_path / "runs")
        make_folder(self.exp_path / "checkpoints")
        make_folder(self.exp_path / "samples")

        self.tbwriter = SummaryWriter(self.exp_path / "runs") if use_tensorboard else None
        self.epoch_stats = None

    def init_stats(self):
        self.epoch_stats = GANEpochLossMeter()

    def update_stats(self, *losses):
        self.epoch_stats.update(*losses)

    def write_stats(self, epoch=None):
        epoch = epoch or self.epoch

        if self.epoch_stats is None:
            raise AttributeError("Epoch statistics non-existant.")
        if self.tbwriter:
            self.tbwriter.add_text("Epoch", str(epoch), epoch)
            self.epoch_stats.write(self.tbwriter, epoch)
        else:
            raise NoTBError

    def save_samples(self, imgs, dest="storage"):
        if dest == "storage":
            imgs_filename = str(self.epoch) + ".png"
            vutils.save_image(imgs, fp=self.exp_path / "samples" / imgs_filename)
        elif dest == "tb":
            if not self.tbwriter:
                raise NoTBError
            img_grid = vutils.make_grid(imgs)
            self.tbwriter.add_image('Generated_samples', img_grid, self.epoch)
        else:
            print("Tried to save images to somewhere else than storage or tensorboard.")

    def save_checkpoint(self, state):
        filename = str(self.epoch) + ".pth"
        path = self.exp_path / "checkpoints" / filename
        chkpt_save(state, path)
        return str(path)

    def load_checkpoint(self, path):
        path = Path(path) # path
        return chkpt_load(path)