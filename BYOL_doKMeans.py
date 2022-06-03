import torch
import numpy as np
from sklearn.cluster import KMeans
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import models
from MY_MODELS import ResNet,BasicBlock,BottleNeck
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from byol_pytorch import BYOL
from torchvision import models
from MY_MODELS import ResNet,BasicBlock,BottleNeck
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

class doKmeans(nn.Module):
    def __init__(self,
                 modelLoadDir,
                 modelLoadNum,
                 clusterNum,
                 maxIter,
                 embedSize,
                 nNeigh,
                 neighThreshold,
                 trnBSize=32,
                 gpuUse=True):
        super(doKmeans,self).__init__()

        self.modelLoadDir = modelLoadDir
        self.modelLoadNum = modelLoadNum
        self.clusterNum = clusterNum
        self.maxIter = maxIter
        self.gpuUse = gpuUse
        self.trnBSize = trnBSize
        self.embedSize = embedSize
        self.nNeigh = nNeigh
        self.neighThreshold = neighThreshold

        self.baseModelBYOL = ResNet(block=BottleNeck,
                                    num_blocks=[3,4,6,3],
                                    num_classes=self.embedSize,
                                    mnst_ver=False)
        print(f'loading {modelLoadDir} {modelLoadNum}')
        modelStateDict = torch.load(self.modelLoadDir+self.modelLoadNum+'.pt')
        self.baseModelBYOL.load_state_dict(modelStateDict)
        print(f'loading {modelLoadDir} {modelLoadNum} successfully')


        transform = transforms.Compose([transforms.ToTensor()])
        self.dataset = CIFAR10(root='~/', train=True, download=True, transform=transform)
        self.trainDataloader = DataLoader(self.dataset,
                                          batch_size=self.trnBSize,
                                          shuffle=False,
                                          num_workers=2)

        if self.gpuUse == True:
            USE_CUDA = torch.cuda.is_available()
            print(USE_CUDA)
            self.device = torch.device('cuda' if USE_CUDA else 'cpu')
            print('학습을 진행하는 기기:', self.device)
        else:
            self.device = torch.device('cpu')
            print('학습을 진행하는 기기:', self.device)


        self.baseModelBYOL.to(self.device)


    def convert2FeatVec(self):

        self.baseModelBYOL.eval()

        with torch.set_grad_enabled(False):

            TDataLoader = tqdm(self.trainDataloader)

            globalTime = time.time()

            totalFeatVecTensor = []
            totalLabelTensor = []

            for idx, (inputs, label) in enumerate(TDataLoader):
                localTime = time.time()

                inputs = inputs.to(self.device)
                eachFeatVecs = self.baseModelBYOL(inputs)
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

        self.baseModelBYOL.train()

        print(f'size of totalFeatVec : {totalFeatVecTensor.size()}'
              f'and size of totalLabel {totalLabelTensor.size()}')

        return totalFeatVecTensor, totalLabelTensor

    def getResultKMeans(self):

        convertedFeatVecs,totalLabelTensor = self.convert2FeatVec()

        print('start clustering')
        eachClusterPredict = KMeans(n_clusters=self.clusterNum,
                                    max_iter=self.maxIter,
                                    verbose=1).fit_predict(convertedFeatVecs)
        print('clustering complete ')

        clusterNum2LabelDict = {}
        for eachCluster in range(self.clusterNum):
            print(f'doing for {eachCluster} th start')
            selectedIdxes = eachClusterPredict == eachCluster

            clutsterMeanLabel = torch.mode(totalLabelTensor[selectedIdxes]).values

            print(f'mode is : {clutsterMeanLabel} and len :{np.sum(selectedIdxes)}')

            clusterNum2LabelDict[eachCluster] = clutsterMeanLabel

        totalPredictedLabels = []
        for eachpredictedClusterNum in eachClusterPredict:
            totalPredictedLabels.append(clusterNum2LabelDict[eachpredictedClusterNum])

        totalPredictedLabels = torch.stack(totalPredictedLabels)
        
        correct = (totalPredictedLabels == totalLabelTensor).float()

        acc = torch.mean(correct)

        neigh = KNeighborsClassifier(n_neighbors=self.nNeigh,
                                     metric='cosine')
        neigh.fit(convertedFeatVecs,totalPredictedLabels)

        eachNeighbors = neigh.kneighbors(convertedFeatVecs,n_neighbors=self.nNeigh)[1]
        eachLabelofNeighs = np.take(totalPredictedLabels,eachNeighbors)

        predLabelArr4Comapre = torch.from_numpy(np.repeat(np.expand_dims(totalPredictedLabels,axis=1),self.nNeigh,axis=1))

        howManyCorrect = (eachLabelofNeighs==predLabelArr4Comapre).float()
        howMeanCorrect = torch.mean(howManyCorrect,dim=1) >= self.neighThreshold


        predWithConfidence = totalPredictedLabels[howMeanCorrect]
        gtWithConfidence = totalLabelTensor[howMeanCorrect]

        labelCompareTensor = (predWithConfidence == gtWithConfidence).float()
        newAcc = torch.mean(labelCompareTensor)

        return acc ,newAcc



modelLoadDir = '/home/a286winteriscoming/'
# modelLoadNum = 'test128_800'
clusterNum = 1000
maxIter = 300
# embedSize = 128

loadNumAndEmbed = [
                   ['test512_600',512]]

for each in loadNumAndEmbed:

    modelLoadNum = each[0]
    embedSize = each[1]

    resultLst = []
    resultLst2 = []

    nNeighLst = [10*(i+1) for i in range(10)]
    neighThresholdLst = [0.91,0.93,0.95,0.97,0.99]

    for nNeigh in nNeighLst:
        for neighThreshold in neighThresholdLst:

            DOIT = doKmeans(modelLoadDir=modelLoadDir,
                            modelLoadNum=modelLoadNum,
                            clusterNum=clusterNum,
                            embedSize=embedSize,
                            nNeigh=nNeigh,
                            neighThreshold=neighThreshold,
                            maxIter=maxIter)

            testAcc,testNewAcc = DOIT.getResultKMeans()
            resultLst.append(testAcc)
            resultLst2.append(testNewAcc)
            print(testAcc,testNewAcc)

    print('-------------------------------')
    print(resultLst)
    print(resultLst2)

    resultDict = {'baselineAcc':resultLst,
                  'newAcc':resultLst2}

    with open(modelLoadDir+modelLoadNum+'_result.pkl','wb') as F:
        pickle.dump(resultDict,F)



        






























