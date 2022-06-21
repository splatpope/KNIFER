from knifer.config import params as P

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms as tr

def base_transforms(img_size):
    return tr.Compose([ 
        tr.Resize(img_size),
        tr.CenterCrop(img_size),
        tr.ToTensor(),
        tr.Normalize(0.5, 0.5),
        #tr.Lambda(lambda t: (t-0.5)*2), # or Normalize(1/2, 1/2) but hey
    ])

class TrainingData():
    def __init__(self, params: P.DatasetParameters):
        import knifer.context as KF
        self.img_size = params.img_size
        #transforms = tr.Compose(params.aug.transforms)
        #transforms = tr.Compose([transforms, base_transforms(params.img_size)])
        transforms = base_transforms(params.img_size)
        self.dataset = ImageFolder(params.path, transform=transforms)
        # + log dset length

        batch_size = KF.UPDATER.batch_size
        self.loader = DataLoader(self.dataset,
            batch_size=batch_size,
            shuffle=True,
        )
