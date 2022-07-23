import numpy as np
from torchvision.datasets import CIFAR10,CIFAR100,STL10
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
import pickle

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
        sample['AugedImage'] = self.augmentation_transform(image)

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



class baseDataset4SCAN(Dataset):
    def __init__(self,
                 downDir,
                 dataType,
                 transform
                 ):
        super(baseDataset4SCAN, self).__init__()

        self.downDir = downDir
        self.dataType = dataType
        self.transform = transform

        if dataType == 'cifar10':
            preDataset = CIFAR10(root=downDir, train=True, download=True)
            self.dataInput = preDataset.data
            self.dataLabel = preDataset.targets
        if dataType == 'cifar100':
            preDataset = CIFAR100(root=downDir, train=True, download=True)
            self.dataInput = preDataset.data
            self.dataLabel = preDataset.targets
        if dataType == 'stl10':
            preDataset = STL10(root=downDir, split='train', download=True)
            self.dataInput = preDataset.data
            self.dataLabel = preDataset.labels

        if dataType in ['imagenet10','imagenet50','imagenet100','imagenet200','tinyImagenet']:
            with open(self.downDir+f'SCAN_imagenets/{dataType}_PathLst.pkl', 'rb') as F:
                self.PathLst = pickle.load(F)

            with open(self.downDir+f'SCAN_imagenets/{dataType}_LabelDict.pkl', 'rb') as F:
                self.labelDict = pickle.load(F)

            self.resize = transforms.Resize((256,256))


    def __len__(self):

        if self.dataType == 'cifar10' or\
                self.dataType == 'cifar100' or\
                self.dataType == 'stl10':
            return len(self.dataInput)
        else:
            return len(self.PathLst)


    def __getitem__(self, idx):

        if self.dataType == 'cifar10' or \
                self.dataType == 'cifar100' or \
                self.dataType == 'stl10':

            img, label = self.dataInput[idx], self.dataLabel[idx]

            if self.dataType=='cifar100':
                label = _cifar100_to_cifar20(label)

            img_size = (img.shape[0],img.shape[1])

            if self.dataType == 'stl10':
                img = Image.fromarray(np.transpose(img, (1, 2, 0)))
                img_size = img.size
            if self.dataType == 'cifar10':
                img = Image.fromarray(img)
            if self.dataType == 'cifar100':
                img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)

            out = {'image': img, 'label': label, 'meta': {'img_size': img_size, 'index': idx}}

            return out

        else:
            path = self.PathLst[idx]
            label = self.labelDict[path.split('/')[-2]]

            with open(path, 'rb') as f:
                img = Image.open(f).convert('RGB')
            img_size = img.size
            
            
            img = self.resize(img)

            if self.transform is not None:
                img = self.transform(img)

            out = {'image': img, 'label': label, 'meta': {'img_size': img_size, 'index': idx}}

            return out



    def get_image(self, index):
        if self.dataType == 'cifar10' or \
                self.dataType == 'cifar100' or \
                self.dataType == 'stl10':

            img = self.data[index]
            return
        else:
            path = self.PathLst[idx]
            label = self.labelDict[path.split('/')[-2]]

            with open(path, 'rb') as f:
                img = Image.open(f).convert('RGB')
            img_size = img.size
            img = self.resize(img)
            return


class filteredDatasetNaive4SCAN(Dataset):
    def __init__(self,
                 downDir,
                 savedIndicesDir,
                 dataType,
                 transform
                 ):
        super(filteredDatasetNaive4SCAN, self).__init__()

        self.downDir = downDir
        self.savedIndicesDir = savedIndicesDir
        self.dataType = dataType
        self.transform = transform

        if dataType == 'cifar10':
            preDataset = CIFAR10(root=downDir, train=True, download=True)
            self.dataInput = preDataset.data
            self.dataLabel = preDataset.targets
        if dataType == 'cifar100':
            preDataset = CIFAR100(root=downDir, train=True, download=True)
            self.dataInput = preDataset.data
            self.dataLabel = preDataset.targets
        if dataType == 'stl10':
            preDataset = STL10(root=downDir, split='train', download=True)
            self.dataInput = preDataset.data
            self.dataLabel = preDataset.labels

        with open(self.savedIndicesDir+'filteredData.pkl','rb') as F:
            self.dataIndices = pickle.load(F)

        if dataType in ['imagenet10', 'imagenet50', 'imagenet100', 'imagenet200', 'tinyImagenet']:
            with open(self.downDir + f'SCAN_imagenets/{dataType}_PathLst.pkl', 'rb') as F:
                self.PathLst = pickle.load(F)

            with open(self.downDir + f'SCAN_imagenets/{dataType}_LabelDict.pkl', 'rb') as F:
                self.labelDict = pickle.load(F)

            self.resize = transforms.Resize((256, 256))

    def __len__(self):

        return len(self.dataIndices['inputIndices'])

    def __getitem__(self, idx):

        if self.dataType == 'cifar10' or \
                self.dataType == 'cifar100' or \
                self.dataType == 'stl10':

            img, label = self.dataInput[self.dataIndices['inputIndices'][idx]], self.dataIndices['pseudoLabels'][idx]

            # if self.dataType == 'cifar100':
            #     label = _cifar100_to_cifar20(label)

            img_size = (img.shape[0], img.shape[1])

            if self.dataType == 'stl10':
                img = Image.fromarray(np.transpose(img, (1, 2, 0)))
                img_size = img.size
            if self.dataType == 'cifar10':
                img = Image.fromarray(img)
            if self.dataType == 'cifar100':
                img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)

            out = {'image': img, 'label': label, 'meta': {'img_size': img_size, 'index': idx}}

            return out

        else:
            path = self.PathLst[self.dataIndices['inputIndices'][idx]]
            label = self.dataIndices['pseudoLabels'][idx]

            with open(path, 'rb') as f:
                img = Image.open(f).convert('RGB')
            img_size = img.size

            img = self.resize(img)

            if self.transform is not None:
                img = self.transform(img)

            out = {'image': img, 'label': label, 'meta': {'img_size': img_size, 'index': idx}}

            return out

    def get_image(self, index):
        if self.dataType == 'cifar10' or \
                self.dataType == 'cifar100' or \
                self.dataType == 'stl10':

            img = self.data[index]
            return
        else:
            path = self.PathLst[idx]
            label = self.labelDict[path.split('/')[-2]]

            with open(path, 'rb') as f:
                img = Image.open(f).convert('RGB')
            img_size = img.size
            img = self.resize(img)
            return


def getCustomizedDataset4SCAN(downDir,dataType,transform,nnNum=None,indices=None,toAgumentedDataset=False,toNeighborDataset=False,baseVer=False):

    dataset = baseDataset4SCAN(downDir=downDir,dataType=dataType,transform=transform)


    if toAgumentedDataset == True:

        dataset = AugmentedDataset(dataset)
        return dataset
    if toNeighborDataset == True:
        thedataset = NeighborsDataset(dataset,indices=indices,nnNum=nnNum)
        return thedataset
    if baseVer == True:
        dataset = dataset

        return dataset



def _cifar100_to_cifar20(target):
  _dict = \
    {0: 4,
     1: 1,
     2: 14,
     3: 8,
     4: 0,
     5: 6,
     6: 7,
     7: 7,
     8: 18,
     9: 3,
     10: 3,
     11: 14,
     12: 9,
     13: 18,
     14: 7,
     15: 11,
     16: 3,
     17: 9,
     18: 7,
     19: 11,
     20: 6,
     21: 11,
     22: 5,
     23: 10,
     24: 7,
     25: 6,
     26: 13,
     27: 15,
     28: 3,
     29: 15,
     30: 0,
     31: 11,
     32: 1,
     33: 10,
     34: 12,
     35: 14,
     36: 16,
     37: 9,
     38: 11,
     39: 5,
     40: 5,
     41: 19,
     42: 8,
     43: 8,
     44: 15,
     45: 13,
     46: 14,
     47: 17,
     48: 18,
     49: 10,
     50: 16,
     51: 4,
     52: 17,
     53: 4,
     54: 2,
     55: 0,
     56: 17,
     57: 4,
     58: 18,
     59: 17,
     60: 10,
     61: 3,
     62: 2,
     63: 12,
     64: 12,
     65: 16,
     66: 12,
     67: 1,
     68: 9,
     69: 19,
     70: 2,
     71: 10,
     72: 0,
     73: 1,
     74: 16,
     75: 12,
     76: 9,
     77: 13,
     78: 15,
     79: 13,
     80: 16,
     81: 19,
     82: 2,
     83: 4,
     84: 6,
     85: 19,
     86: 5,
     87: 5,
     88: 8,
     89: 19,
     90: 18,
     91: 1,
     92: 2,
     93: 15,
     94: 6,
     95: 0,
     96: 17,
     97: 8,
     98: 14,
     99: 13}

  return _dict[target]

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
