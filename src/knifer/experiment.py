from tqdm import tqdm
import torch

import knifer.context as KF
from knifer.config import params as P
from knifer.io.synth import synth_n_fakes

class Experiment():
    def __init__(self, params: P.ExperimentParameters):
        if not KF.is_complete():
            raise ValueError

        self.name = params.name
        self.storage_path = params.storage_path

        self.epoch = 0
        self.fixed_z = torch.randn(32, KF.ARCH.latent_size, 1, 1).to(KF.DEVICE)

    def simple_train_loop(self, n_epochs):

        for _ in range(n_epochs):
            self.epoch += 1
            KF.LOGGER.info(f"Epoch {self.epoch}/{n_epochs}")

            for batch, label in tqdm(KF.DATA.loader, dynamic_ncols=True):
                KF.UPDATER.process_batch(batch, label)

    def train_loop(self, params: P.TrainingParameters):
        KF.ARCH.gen.train()
        KF.ARCH.disc.train()
        start = self.epoch
        for local_step in range(params.n_epochs):
            KF.LOGGER.init_stats()
            special = lambda action: (local_step+1) % action == 0
            self.epoch += 1
            KF.LOGGER.info(f"Epoch {self.epoch}/{start + params.n_epochs}")

            for batch, label in tqdm(KF.DATA.loader, dynamic_ncols=True):
                KF.UPDATER.process_batch(batch, label)
            KF.LOGGER.write_stats(self.epoch)

            if special(params.save_steps):
                self.save()
            if special(params.synth_steps):
                fixed_fakes = synth_n_fakes(z=self.fixed_z)
                KF.LOGGER.save_samples(fixed_fakes, self.epoch, dest="tb")
            if special(params.metrics_steps):
                pass # get fid

    def save(self):
        arch_state = KF.ARCH.serialize()
        upd_state = KF.UPDATER.serialize()
        state = {
            "epoch": self.epoch,
            "arch": arch_state,
            "updater": upd_state,

        }
        KF.LOGGER.save_checkpoint(state, self.epoch)

    
