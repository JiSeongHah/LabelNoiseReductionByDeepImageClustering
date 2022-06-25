import numpy as np
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

        return np.transpose(img,(2,0,1)) , transform1AugedImg, transfrom2AugedImg, label


# from SPICE_CONFIG import Config
# from SPICE_Transformation import get_train_transformations
# import torch
# from torchvision import datasets
# from PIL import Image
# import matplotlib.pyplot as plt
# import random
# from torch.utils.data import DataLoader
#
#
#
# cfg1 =Config.fromfile('/home/a286winteriscoming/PycharmProjects/DATA_VALUATION_REINFORCE/SPICE_Config_cifar10.py')
# trans1 = cfg1.dataConfigs.trans1
# trans2 = cfg1.dataConfigs.trans2
#
# trns1 = get_train_transformations(trans1)
# trns2 = get_train_transformations(trans2)
#
# dt = CustomCifar10(downDir='/home/a286winteriscoming/',
#                    transform1=trns1,
#                    transform2=trns2)
#
# dx = DataLoader(dt,batch_size=32,shuffle=True,num_workers=2)
#
# for ori,trn1,trn2,label in dx:
#     print(ori.size(),trn1.size(),trn2.size(),label.size())
