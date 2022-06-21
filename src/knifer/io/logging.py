
from pathlib import Path
import logging
import json

from torch import save as chkpt_save, load as chkpt_load
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

from knifer.misc_utils import make_folder

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("knifer")
LOGGER.setLevel(logging.INFO)

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


class GANEpochLossMeter():
    def __init__(self):
        self.loss_G = AverageMeter()
        self.loss_D = AverageMeter()
        self.loss_D_real = AverageMeter()
        self.loss_D_fake = AverageMeter()

    @property
    def meters(self) -> list[AverageMeter]:
        return [m for m in vars(self).values()]

    def reset(self):
        [m.reset() for m in self.meters]

    def update(self, vals, n=1):
        if len(vals) != len(self.meters):
            raise ValueError
        [m.update(v,n) for m,v in zip(self.meters, vals)]

    def tbwrite(self, writer:SummaryWriter, epoch):
        loss_dict_D = {
            "D_total": self.loss_D.avg,
            "D_real": self.loss_D_real.avg,
            "D_fake": self.loss_D_fake.avg,
        }
        [writer.add_scalar("Losses/"+k, v, global_step=epoch) for k,v in loss_dict_D.items()]
        writer.add_scalar("Losses/G", self.loss_G.avg, global_step=epoch)

    def __repr__(self):
        return f"Loss G: {self.loss_G.avg} | Loss D (Total/Real/Fake) : {self.loss_D.avg} / {self.loss_D_real.avg} / {self.loss_D_fake.avg}"
        
class GANLogger():
    def __init__(self, experiment, output, use_tensorboard=True):
        self.output_path = Path(output)
        make_folder(output)
        self.experiment = experiment
        self.exp_path = self.output_path / self.experiment
        make_folder(self.exp_path / "runs")
        make_folder(self.exp_path / "checkpoints")
        make_folder(self.exp_path / "samples")

        self.tbwriter = SummaryWriter(self.exp_path / "runs") if use_tensorboard else None
        self.epoch_stats = None

    def info(self, item):
        logger = logging.getLogger("knifer")
        logger.info(item)

    # call this once per run please
    def write_params(self, params: dict, epoch=0):
        if self.tbwriter:
            self.tbwriter.add_text("hyperparameters", json.dumps(params), global_step=epoch)
            self.tbwriter.flush()

    def init_stats(self):
        self.epoch_stats = GANEpochLossMeter()

    def update_stats(self, losses):
        self.epoch_stats.update(losses)

    def write_stats(self, epoch:int):
        if self.epoch_stats is None:
            raise AttributeError("Epoch statistics non-existant.")

        self.info(self.epoch_stats)
        if self.tbwriter:
            self.epoch_stats.tbwrite(self.tbwriter, epoch)
            self.tbwriter.flush()

    def write_FID(self, FID:float, epoch:int):
        self.info(f"FID for epoch {epoch}: {FID}")
        if self.tbwriter:
            self.tbwriter.add_scalar("Performance/FID", FID, epoch)

    def save_samples(self, imgs, epoch:int, dest="storage"):
        if dest == "storage":
            imgs_filename = str(self.epoch) + ".png"
            vutils.save_image(imgs, fp=self.exp_path / "samples" / imgs_filename)
        elif dest == "tb":
            img_grid = vutils.make_grid(imgs)
            self.tbwriter.add_image('Generated_samples', img_grid, global_step=epoch)
            self.tbwriter.flush()
        else:
            print("Tried to save images to somewhere else than storage or tensorboard.")

    def save_checkpoint(self, state:dict, epoch:int) -> str:
        filename = str(epoch) + ".pth"
        path = self.exp_path / "checkpoints" / filename
        chkpt_save(state, path)
        return str(path)

    def load_checkpoint(self, path) -> dict:
        path = Path(path) # path
        return chkpt_load(path)
