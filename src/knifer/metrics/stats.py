from dataclasses import dataclass, asdict

class AverageMeter():
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

@dataclass
class GANEpochLossTag():
    name: str
    value: float

@dataclass
class GANEpochLossReport():
    D_total: float
    D_real: float
    D_fake: float
    G: float

    @property
    def tags(self):
        selfdict = asdict(self)
        return [GANEpochLossTag(f"Losses/{k}", v) for k,v in selfdict.items()]

    def __repr__(self):
        return ' | '.join([f"{t.name}: {t.value}" for t in self.tags])

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

    def report(self):
        return GANEpochLossReport(
            self.loss_D.avg,
            self.loss_D_real.avg,
            self.loss_D_fake.avg,
            self.loss_G.avg,
        )
