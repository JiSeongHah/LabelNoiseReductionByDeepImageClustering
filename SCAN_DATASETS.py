"""
    for code of AugmentedDataset, NeighborsDataset'

    @inproceedings{vangansbeke2020scan,
    title={Scan: Learning to classify images without labels},
    author={Van Gansbeke, Wouter and Vandenhende, Simon and Georgoulis, Stamatios and Proesmans, Marc and Van Gool, Luc},
    booktitle={Proceedings of the European Conference on Computer Vision},
    year={2020}
}
"""

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
import csv





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

        # in the case of imagenet type data
        # pkl containing path of each data and label of each data must be
        # loaded in advance.
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


class noisedDataset4SCAN(Dataset):
    def __init__(self,
                 downDir,
                 savedIndicesDir,
                 dataType,
                 noiseRatio,
                 transform
                 ):
        super(noisedDataset4SCAN, self).__init__()

        self.downDir = downDir
        self.savedIndicesDir = savedIndicesDir
        self.dataType = dataType
        self.noiseRatio = noiseRatio
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


        with open(self.savedIndicesDir + f'noisedDataOnly_{str(self.labelNoiseRatio)}.pkl', 'rb') as F:
            self.noisedDataDict = pickle.load(F)

        if dataType in ['imagenet10', 'imagenet50', 'imagenet100', 'imagenet200', 'tinyImagenet']:
            with open(self.downDir + f'SCAN_imagenets/{dataType}_PathLst.pkl', 'rb') as F:
                self.PathLst = pickle.load(F)

            with open(self.downDir + f'SCAN_imagenets/{dataType}_LabelDict.pkl', 'rb') as F:
                self.labelDict = pickle.load(F)

            self.resize = transforms.Resize((256, 256))

    def __len__(self):

        if self.dataType == 'cifar10' or \
                self.dataType == 'cifar100' or \
                self.dataType == 'stl10':
            return len(self.dataInput)
        else:
            return len(self.PathLst)

    def __getitem__(self, idx):

        if self.dataType == 'cifar10' or \
                self.dataType == 'cifar100' or \
                self.dataType == 'stl10':

            img, label = self.dataInput[idx], self.dataLabel[idx]

            if self.dataType == 'cifar100':
                label = _cifar100_to_cifar20(label)

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





class filteredDatasetNaive4SCAN(Dataset):
    def __init__(self,
                 downDir,
                 savedIndicesDir,
                 dataType,
                 noiseRatio,
                 threshold,
                 transform
                 ):
        super(filteredDatasetNaive4SCAN, self).__init__()

        self.downDir = downDir
        self.savedIndicesDir = savedIndicesDir
        self.dataType = dataType
        self.noiseRatio = noiseRatio
        self.threshold = threshold
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

        with open(self.savedIndicesDir+f'filteredData_{self.threshold}.pkl','rb') as F:
            self.dataIndices = pickle.load(F)

        with open(self.savedIndicesDir+f'cluster2label_{self.noiseRatio}.pkl','rb') as F:
            self.cluster2label = pickle.load(F)

        for k,v in self.cluster2label.items():
            print(f'key : {k}, value : {v}')

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

            img = self.dataInput[self.dataIndices['inputIndices'][idx].item()]
            label = self.cluster2label[self.dataIndices['clusters'][idx].item()]

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
            label = self.dataIndices['clusters'][idx]


            with open(path, 'rb') as f:
                img = Image.open(f).convert('RGB')
            img_size = img.size

            img = self.resize(img)

            if self.transform is not None:
                img = self.transform(img)

            out = {'image': img, 'label': label, 'meta': {'img_size': img_size, 'index': idx}}

            return out



class noisedOnlyDatasetNaive4SCAN(Dataset):
    def __init__(self,
                 downDir,
                 savedIndicesDir,
                 dataType,
                 noiseRatio,
                 transform
                 ):
        super(noisedOnlyDatasetNaive4SCAN, self).__init__()

        self.downDir = downDir
        self.savedIndicesDir = savedIndicesDir
        self.dataType = dataType
        self.noiseRatio = noiseRatio
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

        with open(self.savedIndicesDir+f'noisedDataOnly_{str(self.noiseRatio)}.csv','r') as F:
            rdr = csv.reader(F)
            dataIndices = list(rdr)
            dataIndices = [[int(idx),int(label)] for idx,label in dataIndices]
            self.dataIndices = dataIndices

        if dataType in ['imagenet10', 'imagenet50', 'imagenet100', 'imagenet200', 'tinyImagenet']:
            with open(self.downDir + f'SCAN_imagenets/{dataType}_PathLst.pkl', 'rb') as F:
                self.PathLst = pickle.load(F)

            with open(self.downDir + f'SCAN_imagenets/{dataType}_LabelDict.pkl', 'rb') as F:
                self.labelDict = pickle.load(F)

            self.resize = transforms.Resize((256, 256))

    def __len__(self):


        return len(self.dataIndices)

    def __getitem__(self, idx):

        if self.dataType == 'cifar10' or \
                self.dataType == 'cifar100' or \
                self.dataType == 'stl10':

            img = self.dataInput[self.dataIndices[idx][0]]
            label = self.dataIndices[idx][1]

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
            path = self.PathLst[self.dataIndices[idx][0]]
            label = self.dataIndices[idx][1]

            with open(path, 'rb') as f:
                img = Image.open(f).convert('RGB')
            img_size = img.size

            img = self.resize(img)

            if self.transform is not None:
                img = self.transform(img)

            out = {'image': img, 'label': label, 'meta': {'img_size': img_size, 'index': idx}}

            return out



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
