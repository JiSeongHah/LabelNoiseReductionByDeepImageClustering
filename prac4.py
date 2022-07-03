import torch
import numpy as np
import random
import time
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader,TensorDataset,Dataset
from MY_MODELS import callAnyResnet
import torch.nn.functional as F
import numpy as np


#
# class prac1(Dataset):
#     def __init__(self,
#                  downDir
#                  ):
#         super(prac1, self).__init__()
#
#         self.downDir = downDir
#
#
#
#         preDataset = CIFAR10(root='~/', train=True, download=True)
#
#         self.dataInput = preDataset.data
#         self.dataLabel = preDataset.targets
#
#
#     def __len__(self):
#
#         return 50000
#
#     def __getitem__(self, idx):
#
#         img, label = torch.randn(32,32), torch.randn(1)
#
#         out = {'image': img, 'target': label, 'meta': {'index': idx}}
#
#         return out
#
#     def get_image(self, index):
#         img = self.data[index]
#         return
#
# dt = prac1(downDir='~/')
# ds = DataLoader(dt,batch_size=32,shuffle=False)
#
# for i in ds:
#     print(i['meta']['index'])

class myPrac(torch.nn.Module):
    def __init__(self):
        super(myPrac, self).__init__()

        self.x = []
        print(self.x)

    def doSomething(self):

        plusNum(self.x)

        print(self.x)



def plusNum(inputLst):

    for i in range(10):
        inputLst.append(i**i)


k = myPrac()
k.doSomething()