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
from SCAN_DATASET_CIFAR10 import getCustomizedDataset4SCAN,Cifar104SCAN
from SCAN_trainingProcedure import scanTrain
from SCAN_losses import SCANLoss
import faiss

class doSCAN(nn.Module):
    def __init__(self,
                 modelSaveLoadDir,
                 modelLoadName,
                 headSaveLoadDir,
                 headLoadNum,
                 plotSaveDir,
                 NNSaveDir,
                 embedSize,
                 configPath,
                 clusterNum,
                 nnNum=20,
                 topKNum = 20,
                 downDir='/home/a286/hjs_ver1/mySCAN0/',
                 modelType='resnet18',
                 L2NormalEnd=True,
                 numRepeat=10,
                 entropyWeight=5.0,
                 clusteringWeight=1.0,
                 labelNoiseRatio=0.2,
                 cDim1=512,
                 numHeads=10,
                 reliableCheckNum=100,
                 reliableCheckRatio=0.95,
                 consistencyRatio=0.95,
                 lr=3e-4,
                 wDecay=0,
                 lossMethod='CE',
                 trnBSize=513,
                 filteredTrnBSize=512,
                 valBSize=128,
                 jointTrnBSize=1000,
                 gpuUse=True
                 ):
        super(doSCAN, self).__init__()

        self.modelSaveLoadDir = modelSaveLoadDir
        self.modelLoadName = modelLoadName
        self.headSaveLoadDir = headSaveLoadDir
        self.headLoadNum = headLoadNum
        self.plotSaveDir = plotSaveDir
        createDirectory(self.plotSaveDir)
        self.downDir = downDir
        self.NNSaveDir = NNSaveDir
        self.embedSize = embedSize

        self.modelType = modelType
        self.L2NormalEnd = L2NormalEnd
        self.cDim1 = cDim1
        self.topKNum = topKNum
        self.nnNum= nnNum
        self.configPath = configPath
        self.clusterNum = clusterNum
        self.numHeads = numHeads
        self.labelNoiseRatio = labelNoiseRatio
        self.reliableCheckRatio = reliableCheckRatio
        self.reliableCheckNum = reliableCheckNum
        self.consistencyRatio = consistencyRatio
        self.trnBSize = trnBSize
        self.filteredTrnBSize = filteredTrnBSize
        self.valBSize = valBSize
        self.numRepeat = numRepeat
        self.jointTrnBSize = jointTrnBSize
        self.lossMethod = lossMethod
        self.entropyWeight = entropyWeight
        self.clusteringWeight = clusteringWeight

        dataCfg = Config.fromfile(self.configPath)
        cfgScan = dataCfg.dataConfigs.trans2
        print(cfgScan)
        self.scanTransform = get_train_transformations(cfgScan)

        self.baseTransform = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                       std=[0.229, 0.224, 0.225])])

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

        self.FeatureExtractorBYOL = callAnyResnet(modelType=self.modelType,
                                                  numClass=self.embedSize
                                                  )
        print(f'loading {modelSaveLoadDir} {modelLoadName}')
        modelStateDict = torch.load(self.modelSaveLoadDir + self.modelLoadName)
        self.FeatureExtractorBYOL.load_state_dict(modelStateDict)
        print(f'loading {modelSaveLoadDir}{modelLoadName} successfully')

        self.ClusterHead = myMultiCluster4SCAN(inputDim=self.embedSize,
                                                dim1=self.cDim1,
                                                nClusters=self.clusterNum,
                                                numHead=self.numHeads)

        headLoadedDict= torch.load(self.headSaveLoadDir+str(self.headLoadNum)+'.pt')
        self.ClusterHead.load_state_dict(headLoadedDict)
        print(f'loading saved model : {str(self.headLoadNum)}.pt complete')

        # transform = transforms.Compose([transforms.ToTensor()])
        # self.baseDataset = Cifar104SCAN(downDir='~/', transform1=self.weakAug)
        # self.trainDataloader = DataLoader(self.dataset,
        #                                   batch_size=self.trnBSize,
        #                                   shuffle=True,
        #                                   num_workers=2)
        #
        # self.valDataloader = DataLoader(self.dataset,
        #                                 batch_size=self.valBSize,
        #                                 shuffle=False,
        #                                 num_workers=2)

        self.optimizerBackbone = Adam(self.FeatureExtractorBYOL.parameters(),
                                      lr=self.lr,
                                      eps=1e-9)

        self.optimizerCHead = Adam(self.ClusterHead.parameters(),
                                   lr=self.lr,
                                   eps=1e-9)

        self.headOnlyConsisLossLst = []
        self.headOnlyConsisLossLstAvg = []
        self.headOnlyEntropyLossLst = []
        self.headOnlyEntropyLossLstAvg = []
        self.headOnlyTotalLossLst = []
        self.headOnlyTotalLossLstAvg = []
        self.clusterOnlyAccLst = []

        self.minHeadIdx = 0
        self.minHeadIdxLst = []

        self.clusterOnlyClusteringLossDictPerHead = dict()
        for h in range(self.numHeads):
            self.clusterOnlyClusteringLossDictPerHead[f'head_{h}'] = []

        self.clusterOnlyEntropyLossDictPerHead = dict()
        for h in range(self.numHeads):
            self.clusterOnlyEntropyLossDictPerHead[f'head_{h}'] = []

        self.clusterOnlyTotalLossDictPerHead = dict()
        for h in range(self.numHeads):
            self.clusterOnlyTotalLossDictPerHead[f'head_{h}'] = []

        self.FeatureExtractorBYOL.to(self.device)
        self.ClusterHead.to(self.device)

    def saveNearestNeighbor(self):

        dataset4kNN = getCustomizedDataset4SCAN(downDir=self.downDir,transform=self.baseTransform,baseVer=True)
        dataloader4kNN = DataLoader(dataset4kNN,batch_size=512,shuffle=False,num_workers=2)

        totalFeatures = []
        totalLabels = []
        for idx,i in enumerate(dataloader4kNN):
            inputs = i['image'].to(self.device)
            labels = i['label']
            features = self.FeatureExtractorBYOL(inputs).cpu().clone().detach()
            totalFeatures.append(features)
            totalLabels.append(labels)

        totalFeatures = torch.cat(totalFeatures)
        totalLabels = torch.cat(totalLabels)

        index = faiss.IndexFlatIP(totalFeatures.size(1))
        # index = faiss.index_cpu_to_all_gpus(index)
        index.add(totalFeatures.numpy())
        distances, indices = index.search(totalFeatures.numpy(), self.topKNum+1) # Sample itself is included

        np.save(self.NNSaveDir+'NNs.npy',indices)
        print('saving index of nearest neighbors complete')

    def trainHeadOnly(self,iterNum):
        self.FeatureExtractorBYOL.eval()

        indices = np.load(self.NNSaveDir+'NNs.npy')
        print(123123123123,indices)
        trainDataset = getCustomizedDataset4SCAN(downDir=self.downDir,
                                                 transform=self.scanTransform,
                                                 nnNum= self.nnNum,
                                                 indices=indices,
                                                 toNeighborDataset=True)


        trainDataloader = DataLoader(trainDataset,shuffle=True,batch_size=self.trnBSize,num_workers=2)

        for iter in range(iterNum):
            print(f'{i}/{iterNum} training Start...')
            totalLossDict,\
            consistencyLossDict,\
            entropLossDict = scanTrain(train_loader= trainDataloader,
                                       featureExtractor = self.FeatureExtractorBYOL,
                                       headNum=self.numHeads,
                                       ClusterHead=self.ClusterHead,
                                       criterion=SCANLoss(entropyWeight=self.entropyWeight),
                                       optimizer=self.optimizerCHead,
                                       device=self.device,
                                       update_cluster_head_only=True)

            for h in range(self.numHeads):
                self.clusterOnlyTotalLossDictPerHead[f'head_{h}'].append(np.mean(totalLossDict[f'head_{h}']))
                self.clusterOnlyClusteringLossDictPerHead[f'head_{h}'].append(np.mean(consistencyLossDict[f'head_{h}']))
                self.clusterOnlyEntropyLossDictPerHead[f'head_{h}'].append(np.mean(entropLossDict[f'head_{h}']))
            print(f'{i}/{iterNum} training Complete !!!')

    def valHeadOnly(self):

        self.FeatureExtractorBYOL.eval()
        self.ClusterHead.eval()

        lst4CheckMinLoss = []
        for h in range(self.numHeads):
            lst4CheckMinLoss.append(np.mean(self.clusterOnlyTotalLossDictPerHead[f'head_{h}']))
        print(f'flushing cluster Loss lst complete')

        self.minHeadIdx = np.argmin(lst4CheckMinLoss)
        self.minHeadIdxLst.append(self.minHeadIdx)
        trainDataset = getCustomizedDataset4SCAN(downDir=self.downDir,
                                                 transform=self.baseTransform,
                                                 baseVer=True)

        TDataLoader = tqdm(DataLoader(trainDataset,shuffle=True,batch_size=self.trnBSize,num_workers=2))

        clusterPredResult = []
        gtLabelResult = []
        # predict cluster for each inputs
        for idx, loadedBatch in enumerate(TDataLoader):

            inputsRaw = loadedBatch['image'].float()
            embededInput = self.FeatureExtractorBYOL(inputsRaw.to(self.device))

            clusterProb = self.ClusterHead.forwardWithMinLossHead(embededInput, headIdxWithMinLoss=self.minHeadIdx)
            clusterPred = torch.argmax(clusterProb, dim=1).cpu()
            # print(f'clusterPred hase unique ele : {torch.unique(clusterPred)}')
            clusterPredResult.append(clusterPred)
            gtLabelResult.append(loadedBatch['label'])

        # result of prediction for each inputs
        clusterPredResult = torch.cat(clusterPredResult)
        for eachClusterUnique in torch.unique(clusterPredResult):
            print(
                f' {torch.count_nonzero(clusterPredResult == eachClusterUnique)} allocated for cluster :{eachClusterUnique}'
                f'when validating')

        # ground truth label for each inputs
        gtLabelResult = torch.cat(gtLabelResult).unsqueeze(1)
        print(f'size of gtLabelResult is : {gtLabelResult.size()}')
        # print(f'clusterPred has size : {clusterPredResult.size()} , gtLabelResult has size : {gtLabelResult.size()}')

        ################################# make noised label with ratio ###############################
        ################################# make noised label with ratio ###############################
        minGtLabel = torch.min(torch.unique(gtLabelResult))
        maxGtLabel = torch.max(torch.unique(gtLabelResult))

        # noisedLabels : var for total labels with noised label
        noisedLabels = []
        # noisedLabels4AccCheck : var for checking accruacy of head, contains noised label only
        noisedLabels4AccCheck = []
        # noiseInserTerm : for this term, noised label is inserted into total labels
        noiseInsertTerm = int(1 / self.labelNoiseRatio)
        print(f'noiseInserTerm is : {noiseInsertTerm}')
        for idx, (eachClusterPred, eachGtLabel) in enumerate(zip(clusterPredResult, gtLabelResult)):
            if idx % noiseInsertTerm == 0:
                noisedLabels.append(torch.randint(minGtLabel.cpu(),
                                                  maxGtLabel + 1, (1,)))
                noisedLabels4AccCheck.append([eachClusterPred.cpu(),
                                              eachGtLabel.cpu(),
                                              torch.randint(minGtLabel, maxGtLabel + 1, (1,))])
            else:
                noisedLabels.append(eachGtLabel)
        noisedLabels = torch.cat(noisedLabels)
        print(f'len of noisedLabels4AccCheck is : {len(noisedLabels4AccCheck)}')
        ################################# make noised label with ratio ###############################
        ################################# make noised label with ratio ###############################

        # dict containing which labels is mode in each cluster
        modelLabelPerCluster = dict()
        for eachCluster in range(self.clusterNum):
            sameClusterIdx = clusterPredResult == eachCluster
            # print(torch.sum(sameClusterIdx.float()))
            try:
                modeLabel = torch.mode(noisedLabels[sameClusterIdx]).values
            except:
                modeLabel = torch.tensor([])
            modelLabelPerCluster[eachCluster] = modeLabel
            # print(eachCluster,modelLabelPerCluster[eachCluster])

        accCheck = []
        for eachCheckElement in noisedLabels4AccCheck:
            eachPredictedLabel = eachCheckElement[0].item()
            eachGroundTruthLabel = eachCheckElement[1]
            # if modelLabelPerCluster[eachPredictedLabel].size(0) != 0:

            if modelLabelPerCluster[eachPredictedLabel] == eachGroundTruthLabel:
                accCheck.append(1)
            else:
                accCheck.append(0)

        print(f'len of accCheck is : {len(accCheck)}')

        self.clusterOnlyAccLst.append(np.mean(accCheck))
        print(f'validation step end with accuracy : {np.mean(accCheck)}, total OK is : {np.sum(accCheck)} and '
              f'not OK is : {np.sum(np.array(accCheck) == 0)} with '
              f'length of data : {len(accCheck)}')

    def valHeadOnlyEnd(self):

        self.headOnlyConsisLossLstAvg.append(np.mean(self.clusterOnlyClusteringLossDictPerHead[f'head_{self.minHeadIdx}']))
        self.headOnlyEntropyLossLstAvg.append(
            np.mean(self.clusterOnlyEntropyLossDictPerHead[f'head_{self.minHeadIdx}']))
        self.headOnlyConsisLossLst.clear()
        self.headOnlyEntropyLossLst.clear()

        # for i in self.headOnlyEntropyLossLstAvg:
        #     print('entropy loss for each epoch is : ', i)

        for h in range(self.numHeads):
            self.clusterOnlyTotalLossDictPerHead[f'head_{h}'] = []
            self.clusterOnlyClusteringLossDictPerHead[f'head_{h}'] = []
            self.clusterOnlyEntropyLossDictPerHead[f'head_{h}'] = []

        fig = plt.figure(constrained_layout=True)

        ax1 = fig.add_subplot(1, 3, 1)
        ax1.plot(range(len(self.headOnlyConsisLossLstAvg)), self.headOnlyConsisLossLstAvg)
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('clustering loss')
        # ax1.set_title('Head Only Train Loss clustering')

        ax2 = fig.add_subplot(1, 3, 2)
        ax2.plot(range(len(self.headOnlyEntropyLossLstAvg)), self.headOnlyEntropyLossLstAvg)
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('entropy loss')
        # ax2.set_title('Head Only Train Loss entropy')

        ax3 = fig.add_subplot(1, 3, 3)
        ax3.plot(range(len(self.clusterOnlyAccLst)), self.clusterOnlyAccLst)
        ax3.set_xlabel('epoch')
        ax3.set_ylabel('acc')
        # ax3.set_title(f'val acc , noise ratio : {self.labelNoiseRatio}')

        plt.savefig(self.plotSaveDir + 'HeadOnlyResult.png', dpi=200)
        print('saving head only plot complete !')
        plt.close()
        plt.clf()
        plt.cla()

        with open(self.plotSaveDir+'minLossHeadIdx.pkl','wb') as F:
            pickle.dump(self.minHeadIdxLst,F)
        print('saving head idx of having minimum loss lst')

    def executeTrainingHeadOnly(self,iterNum=1):
        # time.sleep(10)
        self.trainHeadOnly(iterNum=iterNum)
        print('training done')
        # time.sleep(10)
        self.valHeadOnly()
        self.valHeadOnlyEnd()

    def saveHead(self,iteredNum):
        torch.save(self.ClusterHead.state_dict(),self.headSaveLoadDir+str(iteredNum+self.headLoadNum)+'.pt')
        print(f'saving head complete!!!')
        print(f'saving head complete!!!')
        print(f'saving head complete!!!')

    def checkConfidence(self):

        self.FeatureExtractorBYOL.eval()
        self.ClusterHead.eval()

        lst4CheckMinLoss = []
        for h in range(self.numHeads):
            lst4CheckMinLoss.append(np.mean(self.clusterOnlyTotalLossDictPerHead[f'head_{h}']))
        print(f'flushing cluster Loss lst complete')

        self.minHeadIdx = np.argmin(lst4CheckMinLoss)
        trainDataset = getCustomizedDataset4SCAN(downDir=self.downDir,
                                                 transform=self.baseTransform,
                                                 baseVer=True)

        TDataLoader = tqdm(DataLoader(trainDataset,shuffle=True,batch_size=self.trnBSize,num_workers=2))

        clusterPredResult = []
        gtLabelResult = []
        # predict cluster for each inputs
        for idx, loadedBatch in enumerate(TDataLoader):

            inputsRaw = loadedBatch['image'].float()
            embededInput = self.FeatureExtractorBYOL(inputsRaw.to(self.device))

            clusterProb = self.ClusterHead.forwardWithMinLossHead(embededInput, headIdxWithMinLoss=self.minHeadIdx)
            clusterPred = torch.argmax(clusterProb, dim=1).cpu()
            # print(f'clusterPred hase unique ele : {torch.unique(clusterPred)}')
            clusterPredResult.append(clusterPred)
            gtLabelResult.append(loadedBatch['label'])

        # result of prediction for each inputs
        clusterPredResult = torch.cat(clusterPredResult)
        for eachClusterUnique in torch.unique(clusterPredResult):
            print(
                f' {torch.count_nonzero(clusterPredResult == eachClusterUnique)} allocated for cluster :{eachClusterUnique}'
                f'when validating')

        # ground truth label for each inputs
        gtLabelResult = torch.cat(gtLabelResult).unsqueeze(1)
        print(f'size of gtLabelResult is : {gtLabelResult.size()}')
        # print(f'clusterPred has size : {clusterPredResult.size()} , gtLabelResult has size : {gtLabelResult.size()}')

        ################################# make noised label with ratio ###############################
        ################################# make noised label with ratio ###############################
        minGtLabel = torch.min(torch.unique(gtLabelResult))
        maxGtLabel = torch.max(torch.unique(gtLabelResult))

        # noisedLabels : var for total labels with noised label
        noisedLabels = []
        # noisedLabels4AccCheck : var for checking accruacy of head, contains noised label only
        noisedLabels4AccCheck = []
        # noiseInserTerm : for this term, noised label is inserted into total labels
        noiseInsertTerm = int(1 / self.labelNoiseRatio)
        print(f'noiseInserTerm is : {noiseInsertTerm}')
        for idx, (eachClusterPred, eachGtLabel) in enumerate(zip(clusterPredResult, gtLabelResult)):
            if idx % noiseInsertTerm == 0:
                noisedLabels.append(torch.randint(minGtLabel.cpu(),
                                                  maxGtLabel + 1, (1,)))
                noisedLabels4AccCheck.append([eachClusterPred.cpu(),
                                              eachGtLabel.cpu(),
                                              torch.randint(minGtLabel, maxGtLabel + 1, (1,))])
            else:
                noisedLabels.append(eachGtLabel)
        noisedLabels = torch.cat(noisedLabels)
        print(f'len of noisedLabels4AccCheck is : {len(noisedLabels4AccCheck)}')
        ################################# make noised label with ratio ###############################
        ################################# make noised label with ratio ###############################

        # dict containing which labels is mode in each cluster
        modelLabelPerCluster = dict()
        for eachCluster in range(self.clusterNum):
            sameClusterIdx = clusterPredResult == eachCluster
            # print(torch.sum(sameClusterIdx.float()))
            try:
                modeLabel = torch.mode(noisedLabels[sameClusterIdx]).values
            except:
                modeLabel = torch.tensor([])
            modelLabelPerCluster[eachCluster] = modeLabel
            # print(eachCluster,modelLabelPerCluster[eachCluster])

        accCheck = []
        for eachCheckElement in noisedLabels4AccCheck:
            eachPredictedLabel = eachCheckElement[0].item()
            eachGroundTruthLabel = eachCheckElement[1]
            # if modelLabelPerCluster[eachPredictedLabel].size(0) != 0:

            if modelLabelPerCluster[eachPredictedLabel] == eachGroundTruthLabel:
                accCheck.append(1)
            else:
                accCheck.append(0)

        print(f'len of accCheck is : {len(accCheck)}')

        self.clusterOnlyAccLst.append(np.mean(accCheck))
        print(f'validation step end with accuracy : {np.mean(accCheck)}, total OK is : {np.sum(accCheck)} and '
              f'not OK is : {np.sum(np.array(accCheck) == 0)} with '
              f'length of data : {len(accCheck)}')






os.environ['CUDA_VISIBLE_DEVICES'] = "3"
modelLoadDir = '/home/a286winteriscoming/'
modelLoadDir = '/home/a286/hjs_dir1/mySCAN0/'
modelLoadName = 'normalizedVerembSize512'
modelLoadName = 'simclr_cifar-10.pth.tar'
headLoadNum = 2500
embedSize = 128
configPath = '/home/a286/hjs_dir1/mySCAN0/SCAN_Config_cifar10.py'
clusterNum = 10
entropyWeight = 10.0
cDim1 = 128
trnBSize = 512
labelNoiseRatio = 0.2
saveRange= 100


plotsaveName = mk_name(embedSize=embedSize,
                       clusterNum=clusterNum,
                       entropyWeight=entropyWeight,
                       labelNoiseRatio = labelNoiseRatio,
                       cDim1=cDim1
                       )

createDirectory(modelLoadDir + 'dirHeadOnlyTest1/' + plotsaveName
                )

resultSaveDir = modelLoadDir + 'dirHeadOnlyTest1/' + plotsaveName + '/'
headSaveLoadDir = resultSaveDir+'headModels/'
plotSaveDir = resultSaveDir
NNSaveDir = resultSaveDir + 'NNFILE/'
createDirectory(headSaveLoadDir)
createDirectory(NNSaveDir)

do =  doSCAN(modelSaveLoadDir=modelLoadDir,
             modelLoadName=modelLoadName,
             headSaveLoadDir=headSaveLoadDir,
             headLoadNum=headLoadNum,
             plotSaveDir=plotSaveDir,
             NNSaveDir = NNSaveDir,
             embedSize = embedSize,
             cDim1=cDim1,
             labelNoiseRatio = labelNoiseRatio,
             configPath = configPath,
             trnBSize=trnBSize,
             clusterNum = clusterNum)

# do = doSPICE(modelLoadDir=modelLoadDir,
#              modelLoadNum=modelLoadNum,
#              plotSaveDir=modelLoadDir + 'dirHeadOnlyTest1/' + plotsaveName + '/',
#              embedSize=embedSize,
#              trnBSize=1000,
#              filteredTrnBSize=512,
#              numRepeat=numRepeat,
#              gpuUse=True,
#              entropyWeight=entropyWeight,
#              lr=3e-4,
#              cDim1=cDim1,
#              configPath=configPath,
#              clusterNum=clusterNum)

do.saveNearestNeighbor()
for i in range(10000):
    do.executeTrainingHeadOnly()
    if i % saveRange == 0:
        do.saveHead(iteredNum=i)





