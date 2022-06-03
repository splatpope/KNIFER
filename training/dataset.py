import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms as tr
from torch.utils.data import DataLoader

def batch_mean_and_sd(dset):

    loader = DataLoader(dset, batch_size=16, num_workers=2)
    
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in loader:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)        
    return mean,std
  
if __name__ == "__main__":
    tr_combo = tr.Compose([
        tr.Resize(256),
        tr.CenterCrop(256),
        tr.ToTensor(),
    ])

    image_data = ImageFolder("~/Pictures/KNIVES_256_soft", transform=tr_combo)
    mean, std = batch_mean_and_sd(image_data)
    print(mean, std)
    norm = tr.Normalize(mean, std)

    tr_combo_norm = tr.Compose([
        tr_combo,
        norm,
    ])

    img_data_norm = ImageFolder("~/Pictures/KNIVES_256_soft", transform=tr_combo_norm)
    print(batch_mean_and_sd(img_data_norm))
