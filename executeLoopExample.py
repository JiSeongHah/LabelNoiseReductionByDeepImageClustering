import torch
import numpy as np
from sklearn.cluster import KMeans
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import models
from torch.optim import AdamW, Adam, SGD
from MY_MODELS import ResNet, BasicBlock, Bottleneck, callAnyResnet, myCluster4SCAN, myMultiCluster4SCAN
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from byol_pytorch import BYOL
from torchvision import models
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import models
from tqdm import tqdm
import time
from save_funcs import mk_name, createDirectory
import os
from sklearn.neighbors import KNeighborsClassifier
import pickle
from SCAN_Transformation import get_train_transformations
from torch.utils.data import TensorDataset
from SCAN_CONFIG import Config
from SCAN_DATASETS import getCustomizedDataset4SCAN
from SCAN_trainingProcedure import scanTrain
from SCAN_losses import SCANLoss
from SCAN_usefulUtils import getMinHeadIdx,getAccPerConfLst
import faiss
from SCAN_MainLoop import doSCAN



os.environ['CUDA_VISIBLE_DEVICES'] = "0"

baseDir = 'your base directory'
modelLoadDir = 'directory to load saved model'
basemodelLoadDir = 'directory to load pretrained model'
configPath = 'directory to load config'

# 'cifar10' or 'cifar100' or 'stl10' or 'imagenet10'
basemodelLoadName = 'stl10'
# number of loading saved head
headLoadNum = 300
# number of loading saved feature extraction
FELoadNum = 300
# size of feature vector, 512 for cifar10,cifar100,stl10,
# 2048 for imagenet10
embedSize = 512

# number of cluster class
clusterNum = 10

# number of head
numHeads = 10

entropyWeight = 5.0

# size of hidden layer
cDim1 = 512


trnBSize = 128

# noise ratio
labelNoiseRatio = 0.2

# saving interval
saveRange= 20

# layer type of cluster head
# 'linear' or 'mlp'
layerMethod= 'mlp'

# if true:
update_cluster_head_only = True

# deprecated
updateNNTerm = 10

# if true:
# feature vector from feature extractor is l2 normalized
normalizing = False
# if true:
# use linear layer from pretrained model
useLinLayer = False
# if true:
# last output of cluster head is softmaxed
isInputProb = False

# train batch size for step B-2
jointTrnBSize = 128*4

# number of gradient accumulation
accumulNum = 2


# deprecated
nClasss = 10

# deprecated
theNoiseLst = [i/10 for i in range(1,10)]
theNoise= theNoiseLst[0]


plotsaveName = mk_name(embedSize=embedSize,
                       numHeads = numHeads,
                       clusterNum=clusterNum,
                       entropyWeight=entropyWeight,
                       labelNoiseRatio = labelNoiseRatio,
                       cDim1=cDim1,
                       layerMethod=layerMethod,
                       headOnly = update_cluster_head_only,
                       normalizing=normalizing,
                       useLinLayer = useLinLayer,
                       isInputProb=isInputProb
                       )

for IIDX in range(1,2):
    createDirectory(baseDir + f'dirResultImagenet10_new{13}/' + plotsaveName)
    resultSaveDir = baseDir + f'dirResultImagenet10_new{13}/' + plotsaveName + '/'

    headSaveLoadDir = resultSaveDir+'headModels/'
    FESaveLoadDir = resultSaveDir+'FEModels/'
    plotSaveDir = resultSaveDir
    NNSaveDir = resultSaveDir + 'NNFILE/'

    #dirs for further study
    FTedFESaveLoadDir = resultSaveDir + f'FTedFEModels_{theNoise}/'
    # dirs for further study
    FTedheadSaveLoadDir = resultSaveDir + f'FTedHeadModels_{theNoise}/'
    # var for further study
    FTedFELoadNum = 0
    # var for further study
    FTedheadLoadNum = 0

    createDirectory(headSaveLoadDir)
    createDirectory(FESaveLoadDir)
    createDirectory(NNSaveDir)
    createDirectory(FTedFESaveLoadDir)
    createDirectory(FTedheadSaveLoadDir)

    do =  doSCAN(basemodelSaveLoadDir=basemodelLoadDir,
                 basemodelLoadName=basemodelLoadName,
                 headSaveLoadDir=headSaveLoadDir,
                 FESaveLoadDir=FESaveLoadDir,
                 FTedFESaveLoadDir=FTedFESaveLoadDir,
                 FTedheadSaveLoadDir =FTedheadSaveLoadDir,
                 FTedFELoadNum = FTedFELoadNum,
                 FTedheadLoadNum = FTedheadLoadNum,
                 FELoadNum=FELoadNum,
                 headLoadNum=headLoadNum,
                 plotSaveDir=plotSaveDir,
                 NNSaveDir = NNSaveDir,
                 embedSize = embedSize,
                 normalizing=normalizing,
                 useLinLayer = useLinLayer,
                 isInputProb=isInputProb,
                 cDim1=cDim1,
                 numHeads = numHeads,
                 layerMethod=layerMethod,
                 jointTrnBSize= jointTrnBSize,
                 accumulNum = accumulNum,
                 update_cluster_head_only = update_cluster_head_only,
                 labelNoiseRatio = labelNoiseRatio,
                 configPath = configPath,
                 trnBSize=trnBSize,
                 clusterNum = clusterNum)
    #
    # if idx ==0:
    #     do.checkConfidence()
    # do.checkConfidence()


    # do.saveNearestNeighbor()


    # if idx == 0:
    #     do.saveFiltered()
        # for theNoise in theNoiseLst:
        #     do.saveNoiseDataIndices(theNoise)

    # do.loadModel4filtered(nClass=nClasss)
    # for i in range(201):
    #     do.executeFTedTraining(theNoise = theNoise)
    #     if i % saveRange == 0 and i != 0:
    #         do.saveFTedModels(iteredNum=i)



    # to check accuracy per normal data ratio
    # noiseLst = [i/20 for i in range(2,19)]
    # do.checkAccPerNoise(noiseLst)

    # must be executed before training
    do.saveNearestNeighbor()
    for i in range(101):
        do.executeTrainingHeadOnly()
        # do.executeJointTraining()
        if i % saveRange == 0 and i != 0:
            do.saveHead(iteredNum=i)
            do.saveFeatureExtractor(iteredNum=i)
    for i in range(101,301):
        # do.executeTrainingHeadOnly()
        do.executeJointTraining()
        if i % saveRange == 0 and i != 0:
            do.saveHead(iteredNum=i)
            do.saveFeatureExtractor(iteredNum=i)

        # if i % updateNNTerm == 0:
        #     do.saveNearestNeighbor()
        #     print('recalculating NN complete')
        #     print('recalculating NN complete')
        #     print('recalculating NN complete')









