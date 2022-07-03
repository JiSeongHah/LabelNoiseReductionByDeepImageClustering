import numpy as np
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset
from PIL import Image
import os
import math
import numpy as np
import torch
import torchvision.transforms as transforms
import numpy as np
import torch
from torch.utils.data import Dataset


""" 
    AugmentedDataset
    Returns an image together with an augmentation.
"""


class AugmentedDataset(Dataset):
    def __init__(self, dataset):
        super(AugmentedDataset, self).__init__()
        transform = dataset.transform
        dataset.transform = None
        self.dataset = dataset

        if isinstance(transform, dict):
            self.image_transform = transform['standard']
            self.augmentation_transform = transform['augment']

        else:
            self.image_transform = transform
            self.augmentation_transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset.__getitem__(index)
        image = sample['image']

        sample['image'] = self.image_transform(image)
        sample['image_augmented'] = self.augmentation_transform(image)

        return sample


""" 
    NeighborsDataset
    Returns an image with one of its neighbors.
"""


class NeighborsDataset(Dataset):
    def __init__(self, dataset, indices, nnNum=None):
        super(NeighborsDataset, self).__init__()
        transform = dataset.transform

        if isinstance(transform, dict):
            self.anchor_transform = transform['standard']
            self.neighbor_transform = transform['augment']
        else:
            self.anchor_transform = transform
            self.neighbor_transform = transform

        dataset.transform = None
        self.dataset = dataset
        self.indices = indices  # Nearest neighbor indices (np.array  [len(dataset) x k])
        if nnNum is not None:
            self.indices = self.indices[:, :nnNum + 1]
        assert (self.indices.shape[0] == len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        output = {}

        anchor = self.dataset.__getitem__(index)
        neighbor_index = np.random.choice(self.indices[index], 1)[0]

        neighbor = self.dataset.__getitem__(neighbor_index)

        anchor['image'] = self.anchor_transform(anchor['image'])
        neighbor['image'] = self.neighbor_transform(neighbor['image'])

        output['anchor'] = anchor['image']
        output['neighbor'] = neighbor['image']
        output['possible_neighbors'] = torch.from_numpy(self.indices[index])
        output['label'] = anchor['label']

        return output



class Cifar104SCAN(Dataset):
    def __init__(self,
                 downDir,
                 transform):
        super(Cifar104SCAN, self).__init__()

        self.downDir = downDir

        self.transform = transform


        preDataset = CIFAR10(root=downDir, train=True, download=True)

        self.dataInput = preDataset.data
        self.dataLabel = preDataset.targets


    def __len__(self):

        return len(self.dataInput)

    def __getitem__(self, idx):

        img, label = self.dataInput[idx], self.dataLabel[idx]

        img_size = (img.shape[0],img.shape[1])

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'label': label, 'meta': {'img_size': img_size, 'index': idx}}

        return out

    def get_image(self, index):
        img = self.data[index]
        return

def getCustomizedDataset4SCAN(downDir,transform,nnNum=None,indices=None,toAgumentedDataset=False,toNeighborDataset=False,baseVer=False):

    dataset = Cifar104SCAN(downDir=downDir,transform=transform)

    if toAgumentedDataset == True:

        dataset = AugmentedDataset(dataset)
    if toNeighborDataset == True:
        thedataset = NeighborsDataset(dataset,indices=indices,nnNum=nnNum)
        return thedataset
    if baseVer == True:
        dataset = dataset

        return dataset


# thetransform = transforms.Compose([transforms.ToTensor()
#                                 ])
# # dt = Cifar104SCAN(downDir='~/',transform=thetransform)
# dt = getCustomizedDataset4SCAN(downDir='~/',transform=thetransform,toNeighborDataset=True,indices=np.random.randint(20,size=(50000,20)))
# for i in dt:
#     print(i)



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
