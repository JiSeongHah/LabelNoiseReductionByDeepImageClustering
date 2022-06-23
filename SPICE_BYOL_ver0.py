import torch
import numpy as np
from sklearn.cluster import KMeans
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import models
from torch.optim import AdamW,Adam,SGD
from MY_MODELS import ResNet,BasicBlock,BottleNeck
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from byol_pytorch import BYOL
from torchvision import models
from MY_MODELS import ResNet,BasicBlock,BottleNeck,myCluster4SPICE
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import models
from tqdm import tqdm
import time
from save_funcs import mk_name,createDirectory
import os
from sklearn.neighbors import KNeighborsClassifier
import pickle

class doSPICE(nn.Module):
    def __init__(self,
                 modelLoadDir,
                 modelLoadNum,
                 embedSize,
                 cInputDim,
                 cDim1,
                 clusterNum,
                 reliableCheckNum,
                 consistencyNum,
                 weakAug,
                 strongAug,
                 lr=3e-4,
                 wDecay=0,
                 lossMethod = 'CE',
                 trnBSize=32,
                 gpuUse=True):
        super(doSPICE, self).__init__()

        self.modelLoadDir = modelLoadDir
        self.modelLoadNum = modelLoadNum

        self.embedSize = embedSize
        self.cInputDim = cInputDim
        self.cDim1 = cDim1
        self.clusterNum = clusterNum
        self.reliableCheckNum = reliableCheckNum
        self.consistencyNum = consistencyNum
        self.trnBSize = trnBSize
        self.lossMethod = lossMethod

        self.weakAug = weakAug
        self.strongAug =strongAug

        self.lr = lr
        self.wDecay = wDecay
        self.gpuUse = gpuUse
        if self.gpuUse == True:
            USE_CUDA = torch.cuda.is_available()
            print(USE_CUDA)
            self.device = torch.device('cuda' if USE_CUDA else 'cpu')
            print('학습을 진행하는 기기:', self.device)
        else:
            self.device = torch.device('cpu')
            print('학습을 진행하는 기기:', self.device)


        self.FeatureExtractorBYOL = ResNet(block=BottleNeck,
                                    num_blocks=[3,4,6,3],
                                    num_classes=self.embedSize,
                                    mnst_ver=False)
        print(f'loading {modelLoadDir} {modelLoadNum}')
        modelStateDict = torch.load(self.modelLoadDir+self.modelLoadNum+'.pt')
        self.FeatureExtractorBYOL.load_state_dict(modelStateDict)
        print(f'loading {modelLoadDir} {modelLoadNum} successfully')

        self.ClusterHead = myCluster4SPICE(inputDim=cInputDim,
                                            dim1=cDim1,
                                            dim2=cDim2)

        transform = transforms.Compose([transforms.ToTensor()])
        self.dataset = CIFAR10(root='~/', train=True, download=True, transform=transform)
        self.trainDataloader = DataLoader(self.dataset,
                                          batch_size=self.trnBSize,
                                          shuffle=False,
                                          num_workers=2)

        self.optimizerBackbone = Adam(self.FeatureExtractorBYOL.parameters(),
                                      lr=self.lr,
                                      eps=1e-9,
                                      weight_decay=self.wDecay)

        self.optimizerCHead = Adam(self.ClusterHead.parameters(),
                                      lr=self.lr,
                                      eps=1e-9,
                                      weight_decay=self.wDecay)


        self.FeatureExtractorBYOL.to(self.device)
        self.ClusterHead.to(self.device)

    def forwardClusterHead(self,x):

        predBefSoftmax = self.ClusterHead(x)

        predProb = nn.Softmax(predBefSoftmax,dim=1)

        return predProb.cpu().clone().detach()


    def convert2FeatVec(self):

        self.FeatureExtractorBYOL.eval()

        with torch.set_grad_enabled(False):

            TDataLoader = tqdm(self.trainDataloader)

            globalTime = time.time()

            totalFeatVecTensor = []
            totalLabelTensor = []

            for idx, (inputs, label) in enumerate(TDataLoader):
                localTime = time.time()

                inputs = inputs.to(self.device)
                eachFeatVecs = self.FeatureExtractorBYOL(inputs)
                eachFeatVecs = eachFeatVecs.cpu().clone().detach()
                totalFeatVecTensor.append(eachFeatVecs)
                totalLabelTensor.append(label)

                localTimeElaps = round(time.time() - localTime, 2)
                globalTimeElaps = round(time.time() - globalTime, 2)

                TDataLoader.set_description(f'Processing : {idx} / {len(TDataLoader)}')
                TDataLoader.set_postfix(Gelapsed=globalTimeElaps,
                                        Lelapsed=localTimeElaps,
                                        )

        totalFeatVecTensor = torch.cat(totalFeatVecTensor)
        totalLabelTensor = torch.cat(totalLabelTensor)

        self.FeatureExtractorBYOL.train()

        print(f'size of totalFeatVec : {totalFeatVecTensor.size()}'
              f'and size of totalLabel {totalLabelTensor.size()}')

        return totalFeatVecTensor, totalLabelTensor


    def trainClusterLayer(self):

        self.FeatureExtractorBYOL.eval()

        TDataLoader = tqdm(self.trainDataloader)
        globalTime = time.time()

        for idx, (inputs, label) in enumerate(TDataLoader):



            ######################################### E STEP ############################################
            ######################################### E STEP ############################################
            ######################################### E STEP ############################################

            self.ClusterHead.eval()

            # M/K will be selected for topk
            topkNum = int(inputs.size(0)/self.clusterNum)
            localTime = time.time()

            weakAugedInput = self.weakAug(inputs.clone().detach())
            weakAugedInput = weakAugedInput.to(self.device)
            inputs = inputs.to(self.device)


            with torch.set_grad_enabled(False):
                # FB means from First Branch
                # First branch change input to embedding vector
                # eachFeatVecsFB  : (batch size , embeding size)
                eachFeatVecsFB = self.FeatureExtractorBYOL(inputs)
                eachFeatVecsFB = eachFeatVecs.cpu().clone().detach()

                # SB means from Second Branch
                # Second Branch change weakly augmented input to embedding vector
                # eachFeatVecsSB  : (batch size , embedSize)
                eachFeatVecsSB = self.FeatureExtractorBYOL(weakAugedInput)
                eachFeatVecsSB = eachFeatVecSB.cpu().clone().detach()

                #eachProbs : (bach_size, cluster num)
                # probs calculated by embedding vector from second branch
                eachProbsSB = self.forwardClusterHead(eachFeatVecsSB)

            #topkConfidence : (topk num , cluster num)
            topkConfidence = torch.topk(eachProbsSB,dim=0,k=topkNum).indices

            pseudoCentroid = []
            for eachCluster in range(self.clusterNum):
                eachTopK = topkConfidence[:,eachCluster]
                # eachSelectedTensor : totalBatch[idx == topk] for each cluster
                eachSelectedTensor = torch.index_select(input=eachFeatVecsFB,
                                                        dim=0,
                                                        index=eachTopK)
                # sumedTensor : SUM( each selected topk tensor) * K/M
                # sumedTensor : ( embedSize)
                sumedTensor = torch.sum(eachSelectedTensor,dim=0) * (self.clusterNum / inputs.size(0))
                pseudoCentroid.append(sumedTensor)

            pseudoCentroid = torch.stack(pseudoCentroid)
            # pseudoCentroid : (clusterNum , embedSize)

            # To calculate cosSim between embedding vector from first branch
            # and Pseudo Centroid
            # normalizedCentroid : (clusterNum, embedSize)
            normalizedCentroid = F.normalize(pseudoCentroid)
            # normalizedFestFB : (batchSize, embedSize)
            normalizedFeatsFB = F.normalize(eachFeatVecsFB)

            # cosineSim : (batch size , clusterNum)
            cosineSim = F.linear(normalizedFeatsFB,normalizedCentroid)
            topkSim = torch.topk(cosineSim,dim=0,k=topkNum).indices

            # batchPseudoLabel is 2d tensor which element is 1 or 0
            # if data of certain row is topk simliar to certain cluster of certain column
            # then that element is 1. else element is 0
            # batchPseudoLabel : (batch size , cluterNum)
            batchPseudoLabel = torch.zeros_like(cosineSim).scatter(0,topkSim,1)

            # Filter data row which is not belong to any of clusters
            # that data is not trained by algorithm
            check4notTrain = torch.sum(batchPseudoLabel,dim=1)
            batchPseudoLabel = batchPseudoLabel[check4notTrain != 0]
            batchNullPart = (batchPseudoLabel == 0)*(-1e9)

            # finalPseudoLabel : (batch size - filtered num, clutser Num)
            finalPseudoLabel = F.softmax(batchPseudoLabel+batchNullPart,dim=1)
            filteredInput = inputs.cpu().clone().detach()[check4notTrain != 0]


            ######################################### E STEP ############################################
            ######################################### E STEP ############################################
            ######################################### E STEP ############################################

            self.ClusterHead.train()

            strongAugedInput = self.strongAug(filteredInput)

            with torch.set_grad_enabled(True):
                strongAugedInput = strongAugedInput.to(self.device)
                strongAugedFeats = self.FeatureExtractorBYOL(strongAugedInput)

                predProbs = self.ClusterHead(strongAugedFeats)
                predProbs = predProbs.cpu()

                lossDicts = self.ClusterHead.getTotalLoss(x=predProbs,label=finalPseudoLabel)
                lossMean = sum(loss for loss in lossDicts.values())/self.numHead

                self.optimizerCHead.zero_grad()
                lossMean.backward()
                self.optimizerCHead.step()



                localTimeElaps = round(time.time() - localTime, 2)
                globalTimeElaps = round(time.time() - globalTime, 2)

                TDataLoader.set_description(f'Processing : {idx} / {len(TDataLoader)}')
                TDataLoader.set_postfix(Gelapsed=globalTimeElaps,
                                        Lelapsed=localTimeElaps,
                                        )

        self.ClusterHead.eval()

    def trainJointly(self):
        pass
        # self.FeatureExtractorBYOL.eval()
        # self.ClusterHead.eval()
        #
        # TDataLoader = tqdm(self.trainDataloader)
        # for idx, (inputs,labels) in enumerate(TDat  aLoader):
        #     print(faows)
        #
        #

    def testFunc(self):

        for inputs,labels in self.trainDataloader:
            print(inputs.size(),labels)






























































        
        