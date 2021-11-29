from gui import root
import torchvision.datasets as dset
import torchvision.transforms as transforms
import architectures.dcgan.train as dcgan_train
if __name__ == "__main__":
    KNIFER = root.Application("KNIFER")

    knives = dset.ImageFolder("./KNIVES", transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ]))

    arch = dcgan_train.Trainer(knives)
    arch.train(10)
