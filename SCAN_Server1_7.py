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
from SCAN_MainLoop1 import doSCAN



os.environ['CUDA_VISIBLE_DEVICES'] = "7"
baseDir = '/home/a286server2/hjs_dir1/mySCAN0/'
modelLoadDir = '/home/a286winteriscoming/'
basemodelLoadDir = '/home/a286server2/hjs_dir1/mySCAN0/pretrainedModels/'
configPath = '/home/a286server2/hjs_dir1/mySCAN0/SCAN_Configs.py'

basemodelLoadName = 'cifar10'
headLoadNum = 1300
FELoadNum = 1300
embedSize = 512
clusterNum = 10
numHeads = 10
entropyWeight = 5.0
cDim1 = 128
trnBSize = 128
labelNoiseRatio = 0.2
saveRange= 100
layerMethod= 'linear'
update_cluster_head_only = True
updateNNTerm = 10
normalizing = False
useLinLayer = False
isInputProb = False
jointTrnBSize = 4096

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

createDirectory(baseDir + 'dirResult3/' + plotsaveName)
resultSaveDir = baseDir + 'dirResult3/' + plotsaveName + '/'

headSaveLoadDir = resultSaveDir+'headModels/'
FESaveLoadDir = resultSaveDir+'FEModels/'
plotSaveDir = resultSaveDir
NNSaveDir = resultSaveDir + 'NNFILE/'

createDirectory(headSaveLoadDir)
createDirectory(FESaveLoadDir)
createDirectory(NNSaveDir)

do =  doSCAN(basemodelSaveLoadDir=basemodelLoadDir,
             basemodelLoadName=basemodelLoadName,
             headSaveLoadDir=headSaveLoadDir,
             FESaveLoadDir=FESaveLoadDir,
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
             update_cluster_head_only = update_cluster_head_only,
             labelNoiseRatio = labelNoiseRatio,
             configPath = configPath,
             trnBSize=trnBSize,
             clusterNum = clusterNum)


# do.checkConfidence()
do.saveNearestNeighbor()
for i in range(10000):
    do.executeTrainingHeadOnly()
    # do.executeJointTraining()
    if i % saveRange == 0:
        do.saveHead(iteredNum=i)
        do.saveFeatureExtractor(iteredNum=i)

    # if i % updateNNTerm == 0:
    #     do.saveNearestNeighbor()
    #     print('recalculating NN complete')
    #     print('recalculating NN complete')
    #     print('recalculating NN complete')



