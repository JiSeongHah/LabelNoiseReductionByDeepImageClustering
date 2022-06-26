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
from SPICE_Transformation import get_train_transformations
from torch.utils.data import TensorDataset
from SPICE_CONFIG import Config
from SPICE_DATASET_CIFAR10 import CustomCifar10

class doSPICE(nn.Module):
    def __init__(self,
                 modelLoadDir,
                 modelLoadNum,
                 plotSaveDir,
                 embedSize,
                 configPath,
                 clusterNum,
                 labelNoiseRatio=0.2,
                 cDim1=512,
                 reliableCheckNum=100,
                 reliableCheckRatio=0.95,
                 consistencyRatio=0.95,
                 lr=3e-4,
                 wDecay=0,
                 lossMethod = 'CE',
                 trnBSize=50000,
                 filteredTrnBSize=512,
                 valBSize=128,
                 jointTrnBSize=1000,
                 gpuUse=True):
        super(doSPICE, self).__init__()

        self.modelLoadDir = modelLoadDir
        self.modelLoadNum = modelLoadNum
        self.plotSaveDir = plotSaveDir
        createDirectory(self.plotSaveDir)
        self.embedSize = embedSize

        self.cDim1 = cDim1
        self.configPath = configPath
        self.clusterNum = clusterNum
        self.labelNoiseRatio = labelNoiseRatio
        self.reliableCheckRatio = reliableCheckRatio
        self.reliableCheckNum = reliableCheckNum
        self.consistencyRatio = consistencyRatio
        self.trnBSize = trnBSize
        self.filteredTrnBSize = filteredTrnBSize
        self.valBSize = valBSize
        self.jointTrnBSize = jointTrnBSize
        self.lossMethod = lossMethod

        dataCfg =Config.fromfile(self.configPath)
        cfgWeak = dataCfg.dataConfigs.trans1
        cfgStrong = dataCfg.dataConfigs.trans2

        self.weakAug = get_train_transformations(cfgWeak)
        self.strongAug = get_train_transformations(cfgStrong)

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

        self.ClusterHead = myCluster4SPICE(inputDim=self.embedSize,
                                            dim1=cDim1,
                                            clusters=self.clusterNum)

        # transform = transforms.Compose([transforms.ToTensor()])
        self.dataset = CustomCifar10(downDir='~/', transform1=self.weakAug,transform2=self.strongAug)
        self.trainDataloader = DataLoader(self.dataset,
                                          batch_size = self.trnBSize,
                                          shuffle=True,
                                          num_workers=2)

        self.valDataloader = DataLoader(self.dataset,
                                          batch_size=self.valBSize,
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

        self.clusterOnlyAccLst = []
        self.clusterOnlyLossLst = []
        self.clusterOnlyLossLstAvg = []



        self.FeatureExtractorBYOL.to(self.device)
        self.ClusterHead.to(self.device)

    def forwardClusterHead(self,x):


        predBefSoftmax = self.ClusterHead(x)

        SOFTMAX = nn.Softmax(dim=1)

        predProb = SOFTMAX(predBefSoftmax)

        return predProb

    def calLoss(self,logits,labels):
        if self.lossMethod == 'CE':
            LOSS = nn.CrossEntropyLoss()

            preds = torch.argmax(logits,dim=1)

            acc = torch.mean((preds == labels).float())

            return acc, LOSS(logits,labels)


    def trainHeadOnly(self):

        self.FeatureExtractorBYOL.eval()

        TDataLoader = tqdm(self.trainDataloader)
        globalTime = time.time()
        topkNum = int(len(self.dataset) / self.clusterNum)


        totalInputsTrans2 = []
        eachFeatVecsFB = []
        eachFeatvecsSB = []
        eachProbsSBTotal = []

        for idx, (inputsRaw,inputsTrans1,inputsTrans2, label) in enumerate(TDataLoader):

            ######################################### E STEP ############################################
            ######################################### E STEP ############################################
            ######################################### E STEP ############################################

            self.ClusterHead.eval()

            # M/K will be selected for topk

            localTime = time.time()
            inputsRaw = inputsRaw.float()


            with torch.set_grad_enabled(False):
                # FB means from First Branch
                # First branch change input to embedding vector
                # BatchEachFeatVecsFB  : (batch size , embeding size)
                BatchEachFeatVecsFB = self.FeatureExtractorBYOL(inputsRaw.to(self.device))
                BatchEachFeatVecsFB = BatchEachFeatVecsFB.cpu().clone().detach()

                # SB means from Second Branch
                # Second Branch change weakly augmented input to embedding vector
                # BatchEachFeatVecsSB  : (batch size , embedSize)
                BatchEachFeatVecsSB = self.FeatureExtractorBYOL(inputsTrans1.to(self.device))
                # eachFeatVecsSB = eachFeatVecsSB.cpu().clone().detach()

                #eachProbsSB : (bach_size, cluster num)
                # probs calculated by embedding vector from second branch
                eachProbsSB = self.forwardClusterHead(BatchEachFeatVecsSB)

                eachFeatvecsStrongAugedVer = self.FeatureExtractorBYOL(inputsTrans2.to(self.device)).cpu()

                totalInputsTrans2.append(eachFeatvecsStrongAugedVer)
                eachProbsSBTotal.append(eachProbsSB.cpu())
                eachFeatVecsFB.append(BatchEachFeatVecsFB.cpu())
                eachFeatvecsSB.append(BatchEachFeatVecsSB.cpu())

        # del TDataLoader
        eachProbsSBTotal = torch.cat(eachProbsSBTotal)

        for eachUnique in torch.unique(torch.argmax(eachProbsSBTotal,dim=1)):
            print(f'total unique : {torch.unique(torch.argmax(eachProbsSBTotal, dim=1))}')
            print(
                f'{torch.count_nonzero((torch.argmax(eachProbsSBTotal, dim=1) == eachUnique).long())} for batch {eachUnique}')
        print(f'eachProbsSBTotal size is : {eachProbsSBTotal.size()}')
        eachFeatVecsFB = torch.cat(eachFeatVecsFB)
        print(f'eachFeatVecsFB size is : {eachFeatVecsFB.size()}')
        totalInputsTrans2 = torch.cat(totalInputsTrans2)
        print(f'totalInputsTrans2 size is : {totalInputsTrans2.size()}')
        # eachFeatvecsSB = torch.cat(eachFeatvecsSB)

        #topkConfidence : (topk num , cluster num)
        topkConfidence = torch.topk(eachProbsSBTotal,dim=0,k=topkNum).indices
        print(f'topkConfidence size is : {topkConfidence.size()}')


        pseudoCentroid = []
        for eachCluster in range(self.clusterNum):
            eachTopK = topkConfidence[:,eachCluster]
            print(f'eachTOpk size is : {eachTopK.size()}')
            # eachSelectedTensor : totalBatch[idx == topk] for each cluster
            eachSelectedTensor = torch.index_select(input=eachFeatVecsFB,
                                                    dim=0,
                                                    index=eachTopK)
            # sumedTensor : SUM( each selected topk tensor) * K/M
            # sumedTensor : ( embedSize )
            sumedTensor = torch.sum(eachSelectedTensor,dim=0) * (self.clusterNum / len(self.dataset))
            pseudoCentroid.append(sumedTensor)

        pseudoCentroid = torch.stack(pseudoCentroid)
        # print(f'first peudoCentroid is : {pseudoCentroid[0]}')
        print(f'pseudoCentroid size is : {pseudoCentroid.size()}')
        # pseudoCentroid : (clusterNum , embedSize)

        # To calculate cosSim between embedding vector from first branch
        # and Pseudo Centroid
        # normalizedCentroid : (clusterNum, embedSize)
        normalizedCentroid = F.normalize(pseudoCentroid)
        print(f'normalizedCentroid size is : {normalizedCentroid.size()}')
        # normalizedFestFB : (total data size, embedSize)
        normalizedFeatsFB = F.normalize(eachFeatVecsFB)
        print(f'normalizedFeatsFB size is : {normalizedFeatsFB.size()}')

        # cosineSim : (total data size , clusterNum)
        cosineSim = F.linear(normalizedFeatsFB,normalizedCentroid)
        print(f'cosineSime size is : {cosineSim.size()}')
        # topkSim : (topkNum , clusterNum)
        topkSim = torch.topk(cosineSim,dim=0,k=topkNum).indices

        # batchPseudoLabel is 2d tensor which element is 1 or 0
        # if data of certain row is topk simliar to certain cluster of certain column
        # then that element is 1. else element is 0
        # batchPseudoLabel : (total data size , cluterNum)
        PseudoLabel = torch.zeros_like(cosineSim).scatter(0,topkSim,1)
        print(f'Pseudolabels size is : {PseudoLabel.size()}')
        print(f'unique of pseudolabel is : {torch.unique(torch.argmax(PseudoLabel,dim=1))}')
        for eachPseudo in torch.unique(torch.argmax(PseudoLabel,dim=1)):
            print(f'{torch.count_nonzero((torch.argmax(PseudoLabel,dim=1)==eachPseudo).long())} for {eachPseudo}')

        # Filter data row which is not belong to any of clusters
        # that data is not trained by algorithm
        check4notTrain = torch.sum(PseudoLabel,dim=1)
        PseudoLabel = PseudoLabel[check4notTrain != 0]
        NullPart = (PseudoLabel == 0)*(-1e9)


        # finalPseudoLabel : (batch size - filtered num, clutser Num)
        finalPseudoLabel = F.softmax(PseudoLabel+NullPart,dim=1)

        ####################### convert strongAuged input into embedding vector ################
        print('converting strong auged input into embedding vector start')
        # totalInputsTrans2 = []
        # TDataLoader = tqdm(self.trainDataloader)
        # for idx, (inputsRaw, inputsTrans1, inputsTrans2, label) in enumerate(TDataLoader):
        #     inputsTrans2 = inputsTrans2.to(self.device)
        #     eachFeatvecsStrongAugedVer = self.FeatureExtractorBYOL(inputsTrans2)
        #     eachFeatvecsStrongAugedVer = eachFeatvecsStrongAugedVer.cpu()
        #     totalInputsTrans2.append(eachFeatvecsStrongAugedVer)
        # totalInputsTrans2 = torch.cat(totalInputsTrans2)
        print('converting strong auged input into embedding vector complete!!')
        print(f'totalInputsTrans2 size : {totalInputsTrans2.size()} check4noTrain size : {check4notTrain.size()}')

        filteredInput = totalInputsTrans2[check4notTrain != 0].cpu().clone().detach()
        print(f'filtered num is : {torch.sum((check4notTrain!=0)).float()}')
        print(f'filteredInput size is : {filteredInput.size()} and filteredLabel is : {finalPseudoLabel.size()}')


        ######################################### E STEP ############################################
        ######################################### E STEP ############################################
        ######################################### E STEP ############################################

        self.ClusterHead.train()
        self.optimizerCHead.zero_grad()
        DatasetFromFilteredData = TensorDataset(filteredInput,finalPseudoLabel)
        DataloaderFromFilteredData = tqdm(DataLoader(DatasetFromFilteredData,batch_size=self.filteredTrnBSize,shuffle=True,num_workers=2))

        gradientStep = len(DataloaderFromFilteredData)

        with torch.set_grad_enabled(True):
            for idx, (theInputs,theLabels) in enumerate(DataloaderFromFilteredData):

                SOFTMAX = nn.Softmax(dim=1)

                theInputs = theInputs.to(self.device)
                predProbs = self.forwardClusterHead(theInputs)
                predProbs = SOFTMAX(predProbs.cpu())
                # print(torch.max(predProbs),torch.argmax(predProbs,dim=1))

                lossResult = self.ClusterHead.getLoss(pred=predProbs,label=theLabels)/gradientStep
                # lossMean = sum(loss for loss in lossDicts.values())/self.numHead
                lossResult.backward()

                self.clusterOnlyLossLst.append(lossResult.item())

                localTimeElaps = round(time.time() - localTime, 2)
                globalTimeElaps = round(time.time() - globalTime, 2)

                TDataLoader.set_description(f'Processing : {idx} / {len(TDataLoader)}')
                TDataLoader.set_postfix(Gelapsed=globalTimeElaps,
                                        Lelapsed=localTimeElaps,
                                        LOSS=lossResult.item())

            self.optimizerCHead.step()
            self.optimizerCHead.zero_grad()
        # print(f'eachFeatVecFB is in device : {eachFeatVecsFB.device}')
        # print(f'eachFeatVecSB is in device : {eachFeatVecsSB.device}')
        # print(f'eachProbsSB is in device : {eachProbsSB.device}')
        # print(f'topkConfidence is in device : {topkConfidence.device}')
        # print(f'pseudoCentroid is in device : {pseudoCentroid.device}')
        # print(f'normalizedCentroid is in : {normalizedCentroid.device}')
        # print(f'normalizedFeatsFB is in {normalizedFeatsFB.device}')
        # print(f'cosineSim is in :{cosineSim.device}')
        # print(f'topkSim is in : {topkSim.device}')
        # print(f'inputsRaw is in : {inputsRaw.device}')
        # print(f'inputstrans1 is in : {inputsTrans1.device}')
        # print(f'inputstrans2 is in : {inputsTrans2.device}')
        #
        # del predProbs
        # del eachProbsSB
        # del topkConfidence
        # del filteredInput
        # del eachFeatVecsFB
        # del eachFeatVecsSB
        # del strongAugedFeats
        # del TDataLoader

        self.ClusterHead.eval()

    def validationHeadOnly(self):

        self.FeatureExtractorBYOL.eval()
        self.ClusterHead.eval()

        TDataLoader = tqdm(self.trainDataloader)

        clusterPredResult = []
        gtLabelResult = []

        # predict cluster for each inputs
        for idx, (inputsRaw,inputsTrans1,inputsTrans2, label) in enumerate(TDataLoader):

            inputsRaw=inputsRaw.float()
            embededInput = self.FeatureExtractorBYOL(inputsRaw.to(self.device))

            clusterProb = self.forwardClusterHead(embededInput)
            clusterPred = torch.argmax(clusterProb,dim=1)
            # print(f'clusterPred hase unique ele : {torch.unique(clusterPred)}')
            clusterPredResult.append(clusterPred)
            gtLabelResult.append(label)

        # result of prediction for each inputs
        clusterPredResult =torch.cat(clusterPredResult)
        # ground truth label for each inputs
        gtLabelResult = torch.cat(gtLabelResult).unsqueeze(1)
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
        noiseInsertTerm = int(self.labelNoiseRatio*len(gtLabelResult))
        for idx,(eachClusterPred, eachGtLabel) in enumerate(zip(clusterPredResult,gtLabelResult)):
            if idx % noiseInsertTerm == 0:
                noisedLabels.append(torch.randint(minGtLabel.cpu(),
                                                  maxGtLabel+1,(1,)))
                noisedLabels4AccCheck.append([eachClusterPred.cpu(),
                                              eachGtLabel.cpu(),
                                              torch.randint(minGtLabel,maxGtLabel+1,(1,))])
            else:
                noisedLabels.append(eachGtLabel)
        noisedLabels = torch.cat(noisedLabels)
        ################################# make noised label with ratio ###############################
        ################################# make noised label with ratio ###############################

        # dict containing which labels is mode in each cluster
        modelLabelPerCluster= dict()
        for eachCluster in range(self.clusterNum):
            sameClusterIdx = clusterPredResult == eachCluster
            print(torch.sum(sameClusterIdx.float()))
            try:
                modeLabel = torch.mode(noisedLabels[sameClusterIdx]).values
            except:
                modeLabel = torch.tensor([])
            modelLabelPerCluster[eachCluster] = modeLabel
            print(eachCluster,modelLabelPerCluster[eachCluster])

        accCheck = []
        for eachCheckElement in noisedLabels4AccCheck:
            eachPredictedLabel = eachCheckElement[0].item()
            eachGroundTruthLabel = eachCheckElement[1]
            # if modelLabelPerCluster[eachPredictedLabel].size(0) != 0:

            if modelLabelPerCluster[eachPredictedLabel] == eachGroundTruthLabel:
                accCheck.append(1)
            else:
                accCheck.append(0)


        self.clusterOnlyAccLst.append(np.mean(accCheck))
        print(f'validation step end with accuracy : {np.mean(accCheck)}')


    def validationHeadOnlyEnd(self):

        self.clusterOnlyLossLstAvg.append(np.mean(self.clusterOnlyLossLst))
        self.clusterOnlyLossLst.clear()

        fig = plt.figure(constrained_layout=True)

        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(range(len(self.clusterOnlyLossLstAvg)), self.clusterOnlyLossLstAvg)
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss')
        ax1.set_title('Head Only Train Loss')

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(range(len(self.clusterOnlyAccLst)), self.clusterOnlyAccLst)
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('acc')
        ax2.set_title(f'val acc , noise ratio : {self.labelNoiseRatio}')

        plt.savefig(self.plotSaveDir+'HeadOnlyResult.png',dpi=200)
        print('saving head only plot complete !')
        plt.close()
        plt.clf()
        plt.cla()

    def executeTrainingHeadOnly(self):\

        # time.sleep(10)
        self.trainHeadOnly()
        print('training done')
        # time.sleep(10)
        self.validationHeadOnly()
        self.validationHeadOnlyEnd()


    def trainJointly(self):

        self.ClusterHead.eval()

        totalReliableInput = []
        totalLabelForReliableInput = []
        totalUnReliableInput = []

        with torch.set_grad_enabled(False):

            TDataLoader = tqdm(self.trainDataloader)
            globalTime = time.time()

            for idx, (inputs, label) in enumerate(TDataLoader):

                ######################################### E STEP ############################################
                ######################################### E STEP ############################################
                ######################################### E STEP ############################################

                self.ClusterHead.eval()

                # M/K will be selected for topk
                topkNum = int(inputs.size(0) / self.clusterNum)
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

                    # eachProbs : (bach_size, cluster num)
                    # probs calculated by embedding vector from second branch
                    eachProbsSB = self.forwardClusterHead(eachFeatVecsSB)

                # topkConfidence : (topk num , cluster num)
                topkConfidence = torch.topk(eachProbsSB, dim=0, k=topkNum).indices

                pseudoCentroid = []
                for eachCluster in range(self.clusterNum):
                    eachTopK = topkConfidence[:, eachCluster]
                    # eachSelectedTensor : totalBatch[idx == topk] for each cluster
                    eachSelectedTensor = torch.index_select(input=eachFeatVecsFB,
                                                            dim=0,
                                                            index=eachTopK)
                    # sumedTensor : SUM( each selected topk tensor) * K/M
                    # sumedTensor : ( embedSize)
                    sumedTensor = torch.sum(eachSelectedTensor, dim=0) * (self.clusterNum / inputs.size(0))
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
                cosineSim = F.linear(normalizedFeatsFB, normalizedCentroid)
                topkSim = torch.topk(cosineSim, dim=0, k=topkNum).indices

                # batchPseudoLabel is 2d tensor which element is 1 or 0
                # if data of certain row is topk simliar to certain cluster of certain column
                # then that element is 1. else element is 0
                # batchPseudoLabel : (batch size , cluterNum)
                batchPseudoLabel = torch.zeros_like(cosineSim).scatter(0, topkSim, 1)

                # Filter data row which is not belong to any of clusters
                # that data is not trained by algorithm
                check4notTrain = torch.sum(batchPseudoLabel, dim=1)
                batchPseudoLabel = batchPseudoLabel[check4notTrain != 0]
                batchNullPart = (batchPseudoLabel == 0) * (-1e9)

                # finalPseudoLabel : (batch size - filtered num, clutser Num)
                finalPseudoLabel = F.softmax(batchPseudoLabel + batchNullPart, dim=1)
                ReliableInput = inputs.cpu().clone().detach()[check4notTrain != 0]
                UnReliableInput = inputs.cpu().clone().detach()[check4notTrain == 0]

                totalReliableInput.append(ReliableInput)
                totalLabelForReliableInput.append(finalPseudoLabel)
                totalUnReliableInput.append(UnReliableInput)

                ######################################### E STEP ############################################
                ######################################### E STEP ############################################
                ######################################### E STEP ############################################

        # totalReliableInput : ( filtered data size , embedding size)
        totalReliableInput = torch.cat(totalReliableInput)
        # totalLabelForReliableInput : ( data size, 1)
        totalLabelForReliableInput = torch.cat(totalLabelForReliableInput)
        # totalUnReliableInput : ( total data size - filtered data size , embedding size)
        totalUnReliableInput = torch.cat(totalUnReliableInput)
        # cosSim4RelInput : ( filtered data size, filtered data size)
        cosSim4RelInput = F.linear(F.normalize(totalReliableInput),F.normalize(totalReliableInput))
        # topKNeighbors : (filtered data size, topkNum+1), start from 1 to filter data itself
        topKNeighbors = torch.topk(cosSim4RelInput,dim=1,k=self.reliableCheckNum+1).indices[:,1:]


        finalReliableInput = []
        finalReliableLabel =[]

        finalSubReliableInput = []
        finalSubReliableLabel = []

        for eachInput, eachRow,eachLabel in zip(totalReliableInput,topKNeighbors,totalLabelForReliableInput):
            # check how many neighbors have same label
            labelOfNeighbors = torch.index_select(totalLabelForReliableInput,0,eachRow) == eachLabel
            labelOfNeighbors = torch.mean(labelOfNeighbors.float()) > self.reliableCheckRatio

            if labelOfNeighbors == True:
                finalReliableInput.append(eachInput)
                finalReliableLabel.append(eachLabel)
            else:
                probOfWeakAug = self.ClusterHead(self.weakAug(eachInput))
                if torch.max(probOfWeakAug) > self.consistencyRatio:
                    finalSubReliableInput.append(eachInput)
                    finalSubReliableLabel.append(torch.argmax(probOfWeakAug,dim=1))
                else:
                    pass

        for eachInput in totalUnReliableInput:
            probOfWeakAug = self.ClusterHead(self.weakAug(eachInput))
            if torch.max(probOfWeakAug)> self.consistencyRatio:
                finalSubReliableInput.append(eachInput)
                finalSubReliableLabel.append(torch.argmax(probOfWeakAug,dim=1))
            else:
                pass


        finalReliableInput = torch.cat(finalReliableInput)
        finalReliableLabel = torch.cat(finalSubReliableLabel)
        ReliableInputIdx = torch.ones(len(finalReliableInput))

        finalSubReliableInput = torch.cat(finalSubReliableInput)
        finalSubReliableLabel = torch.cat(finalSubReliableLabel)
        SubReliableInputIdx = torch.zeros(len(finalSubReliableInput))

        finalTotalTrnInput = torch.cat([finalReliableInput,finalSubReliableInput])
        finalTotalTrnLabel = torch.cat([finalReliableLabel,finalSubReliableLabel])
        finatlIdx = torch.cat([ReliableInputIdx,SubReliableInputIdx])

        theDataset = TensorDataset(finalTotalTrnInput,finalTotalTrnLabel,finatlIdx)
        theDataloader = DataLoader(theDataset,batch_size=self.jointTrnBSize,shuffle=True)

        self.FeatureExtractorBYOL.train()
        self.ClusterHead.train()
        with torch.set_grad_enabled(True):
            for theInputs,theLabels,theIdxes in theDataloader:

                reliableIdx = idxes == 1
                unReliableIdx = idxes == 0

                if torch.sum(reliableIdx.float()) != 0 and torch.sum(unReliableIdx.float()) != 0:

                    reliableInput = self.weakAug(theInputs[reliableIdx])
                    reliableLabel = theLabels[reliableIdx]

                    unReliableInput = self.strongAug(theInputs[unReliableIdx])
                    unReliableLabel = theLabels[unReliableIdx]

                    realiableLogits = self.forwardClusterHead(self.FeatureExtractorBYOL(reliableInput))
                    realiableLoss = self.calLoss(realiableLogits,reliableLabel)

                    unReliableLogits = self.forwardClusterHead(self.FeatureExtractorBYOL(unReliableInput))
                    unReliableLoss = self.calLoss(unReliableLogits,unReliableLabel)

                    totalLoss = realiableLoss+unReliableLoss
                    self.optimizerBackbone.zero_grad()
                    self.optimizerCHead.zero_grad()
                    totalLoss.backward()
                    self.optimizerBackbone.step()
                    self.optimizerCHead.step()


        self.FeatureExtractorBYOL.eval()
        self.ClusterHead.eval()


modelLoadDir = '/home/a286winteriscoming/'
modelLoadDir = '/home/a286/hjs_dir1/mySPICE0/'
modelLoadNum = 'test2000'
embedSize = 256
configPath = '/home/a286/hjs_dir1/mySPICE0/SPICE_Config_cifar10.py'
clusterNum = 100


do = doSPICE(modelLoadDir=modelLoadDir,
             modelLoadNum=modelLoadNum,
             plotSaveDir=modelLoadDir+'dirHeadOnlyTest1/',
             embedSize=embedSize,
             trnBSize=1000,
             filteredTrnBSize=512,
             gpuUse=True,
             lr=0.005,
             cDim1=256,
             configPath=configPath,
             clusterNum=clusterNum)

for i in range(10000):
    do.executeTrainingHeadOnly()































































        
        