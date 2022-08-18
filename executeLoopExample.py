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
baseDir = '/home/a286/hjs_dir1/mySCAN1/'
modelLoadDir = '/home/a286winteriscoming/'
basemodelLoadDir = '/home/a286/hjs_dir1/mySCAN1/pretrainedModels/'
configPath = '/home/a286/hjs_dir1/mySCAN1/SCAN_Configs.py'
#
# baseDir = 'your base directory'
# modelLoadDir = 'directory to load saved model'
# basemodelLoadDir = 'directory to load pretrained model'
# configPath = 'directory to load config'

basemodelLoadName = 'stl10'
headLoadNum = 300
FELoadNum = 300
embedSize = 512
clusterNum = 10
numHeads = 10
entropyWeight = 5.0
cDim1 = 512
trnBSize = 128
labelNoiseRatio = 0.2
saveRange= 20
layerMethod= 'mlp'
update_cluster_head_only = True
updateNNTerm = 10
normalizing = False
useLinLayer = False
isInputProb = False
jointTrnBSize = 128*4
accumulNum = 2

nClasss = 10
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

    FTedFESaveLoadDir = resultSaveDir + f'FTedFEModels_{theNoise}/'
    FTedheadSaveLoadDir = resultSaveDir + f'FTedHeadModels_{theNoise}/'
    FTedFELoadNum = 0
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



    #
    # noiseLst = [i/20 for i in range(2,19)]
    # do.checkAccPerNoise(noiseLst)
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









    # for idx, theNoise in enumerate(theNoiseLst):
    #
    #     plotsaveName = mk_name(embedSize=embedSize,
    #                            numHeads = numHeads,
    #                            clusterNum=clusterNum,
    #                            entropyWeight=entropyWeight,
    #                            labelNoiseRatio = labelNoiseRatio,
    #                            cDim1=cDim1,
    #                            layerMethod=layerMethod,
    #                            # headOnly = update_cluster_head_only,
    #                            normalizing=normalizing,
    #                            useLinLayer = useLinLayer,
    #                            isInputProb=isInputProb
    #                            )
    #
    #     createDirectory(baseDir + 'dirResultImagenet10_new0/' + plotsaveName)
    #     resultSaveDir = baseDir + 'dirResultImagenet10_new0/' + plotsaveName + '/'
    #
    #     headSaveLoadDir = resultSaveDir+'headModels/'
    #     FESaveLoadDir = resultSaveDir+'FEModels/'
    #     plotSaveDir = resultSaveDir
    #     NNSaveDir = resultSaveDir + 'NNFILE/'
    #
    #     FTedFESaveLoadDir = resultSaveDir + f'FTedFEModels_{theNoise}/'
    #     FTedheadSaveLoadDir = resultSaveDir + f'FTedHeadModels_{theNoise}/'
    #     FTedFELoadNum = 0
    #     FTedheadLoadNum = 0
    #
    #     createDirectory(headSaveLoadDir)
    #     createDirectory(FESaveLoadDir)
    #     createDirectory(NNSaveDir)
    #     createDirectory(FTedFESaveLoadDir)
    #     createDirectory(FTedheadSaveLoadDir)
    #
    #     do =  doSCAN(basemodelSaveLoadDir=basemodelLoadDir,
    #                  basemodelLoadName=basemodelLoadName,
    #                  headSaveLoadDir=headSaveLoadDir,
    #                  FESaveLoadDir=FESaveLoadDir,
    #                  FTedFESaveLoadDir=FTedFESaveLoadDir,
    #                  FTedheadSaveLoadDir =FTedheadSaveLoadDir,
    #                  FTedFELoadNum = FTedFELoadNum,
    #                  FTedheadLoadNum = FTedheadLoadNum,
    #                  FELoadNum=FELoadNum,
    #                  headLoadNum=headLoadNum,
    #                  plotSaveDir=plotSaveDir,
    #                  NNSaveDir = NNSaveDir,
    #                  embedSize = embedSize,
    #                  normalizing=normalizing,
    #                  useLinLayer = useLinLayer,
    #                  isInputProb=isInputProb,
    #                  cDim1=cDim1,
    #                  numHeads = numHeads,
    #                  layerMethod=layerMethod,
    #                  jointTrnBSize= jointTrnBSize,
    #                  accumulNum = accumulNum,
    #                  update_cluster_head_only = update_cluster_head_only,
    #                  labelNoiseRatio = labelNoiseRatio,
    #                  configPath = configPath,
    #                  trnBSize=trnBSize,
    #                  clusterNum = clusterNum)
    #
    #     if idx ==0:
    #         do.checkConfidence()
    #     # do.checkConfidence()
    #     # do.saveNearestNeighbor()
    #     if idx == 0:
    #         do.saveFiltered()
    #         # for theNoise in theNoiseLst:
    #         #     do.saveNoiseDataIndices(theNoise)
    #
    #     do.loadModel4filtered(nClass=nClasss)
    #     for i in range(201):
    #         do.executeFTedTraining(theNoise = theNoise)
    #         if i % saveRange == 0 and i != 0:
    #             do.saveFTedModels(iteredNum=i)
    #
    #     #
    #     # for i in range(201):
    #     #     # do.executeTrainingHeadOnly()
    #     #     do.executeJointTraining()
    #     #     if i % saveRange == 0 and i != 0:
    #     #         do.saveHead(iteredNum=i)
    #     #         do.saveFeatureExtractor(iteredNum=i)
    #
    #         # if i % updateNNTerm == 0:
    #         #     do.saveNearestNeighbor()
    #         #     print('recalculating NN complete')
    #         #     print('recalculating NN complete')
    #         #     print('recalculating NN complete')
    #
    #
    #
    #
