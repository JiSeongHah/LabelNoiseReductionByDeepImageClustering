from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image


class CustomCifar10(Dataset):
    def __init__(self,
                 downDir,
                 transform1,
                 transform2):
        super(CustomCifar10, self).__init__()

        self.downDir = downDir

        self.transform1 = transform1
        self.transform2 = transform2

        preDataset = CIFAR10(root='~/', train=True, download=True)

        self.dataInput = preDataset.data
        self.dataLabel = preDataset.targets


    def __len__(self):

        return len(self.dataInput)

    def __getitem__(self, idx):

        img, label = self.dataInput[idx], self.dataLabel[idx]

        imgPIL = Image.fromarray(img)

        transform1AugedImg = self.transform1(imgPIL)

        transfrom2AugedImg = self.transform2(imgPIL)

        return img , transform1AugedImg, transfrom2AugedImg, label
