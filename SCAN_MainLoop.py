import csv
import torch
import numpy as np
from sklearn.cluster import KMeans
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import models
from torch.optim import AdamW, Adam, SGD
from MY_MODELS import ResNet, BasicBlock, Bottleneck, callAnyResnet, myCluster4SCAN, myMultiCluster4SCAN,myPredictorHead
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
from SCAN_DATASETS import getCustomizedDataset4SCAN,filteredDatasetNaive4SCAN,noisedOnlyDatasetNaive4SCAN
from SCAN_trainingProcedure import scanTrain,selflabelTrain,trainWithFiltered
from SCAN_losses import SCANLoss,selfLabelLoss,filteredDataLoss
from SCAN_usefulUtils import getMinHeadIdx,getAccPerConfLst,loadPretrained4imagenet,Pseudo2Label
import faiss
from torch.nn import DataParallel

class doSCAN(nn.Module):
    def __init__(self,
                 basemodelSaveLoadDir,
                 basemodelLoadName,
                 headSaveLoadDir,
                 FESaveLoadDir,
                 FELoadNum,
                 FTedFESaveLoadDir,
                 FTedheadSaveLoadDir,
                 FTedFELoadNum,
                 FTedheadLoadNum,
                 headLoadNum,
                 plotSaveDir,
                 NNSaveDir,
                 embedSize,
                 configPath,
                 clusterNum,
                 normalizing,
                 useLinLayer,
                 isInputProb,
                 accumulNum=4,
                 update_cluster_head_only=False,
                 layerMethod='linear',
                 nnNum=20,
                 topKNum = 50,
                 selfLabelThreshold=0.99,
                 downDir='your directory to download data',
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
                 trnBSize=128,
                 valBSize=128,
                 jointTrnBSize=1000,
                 gpuUse=True
                 ):
        super(doSCAN, self).__init__()

        self.basemodelSaveLoadDir = basemodelSaveLoadDir

        # to load pretrained model.
        if basemodelLoadName == 'cifar10':
            self.basemodelLoadName = 'simclr_cifar-10.pth.tar'
            self.dataType= basemodelLoadName
        if basemodelLoadName == 'cifar100':
            self.basemodelLoadName = 'simclr_cifar-20.pth.tar'
            self.dataType = basemodelLoadName
        if basemodelLoadName == 'stl10':
            self.basemodelLoadName = 'simclr_stl-10.pth.tar'
            self.dataType = basemodelLoadName
        if basemodelLoadName == 'imagenet10':
            self.basemodelLoadName = 'moco_v2_800ep_pretrain.pth.tar'
            self.dataType = basemodelLoadName
        if basemodelLoadName == 'imagenet50':
            self.basemodelLoadName = 'moco_v2_800ep_pretrain.pth.tar'
            self.dataType = basemodelLoadName
        if basemodelLoadName == 'imagenet100':
            self.basemodelLoadName = 'moco_v2_800ep_pretrain.pth.tar'
            self.dataType = basemodelLoadName
        if basemodelLoadName == 'imagenet200':
            self.basemodelLoadName = 'moco_v2_800ep_pretrain.pth.tar'
            self.dataType = basemodelLoadName
        if basemodelLoadName == 'tinyimagenet':
            self.basemodelLoadName = 'moco_v2_800ep_pretrain.pth.tar'
            self.dataType = basemodelLoadName

        # trained cluster head save dir
        self.headSaveLoadDir = headSaveLoadDir
        # trained feature extractor save dir
        self.FESaveLoadDir = FESaveLoadDir
        # feature extractor load number
        self.FELoadNum = FELoadNum
        # cluster head load number
        self.headLoadNum = headLoadNum

        self.FTedFESaveLoadDir = FTedFESaveLoadDir
        self.FTedheadSaveLoadDir = FTedheadSaveLoadDir
        self.FTedFELoadNum = FTedFELoadNum
        self.FTedheadLoadNum = FTedheadLoadNum

        # dir for saving result plot
        self.plotSaveDir = plotSaveDir
        createDirectory(self.plotSaveDir)

        # dir for downloading data
        self.downDir = downDir
        # dir for saving indices of nearest neighbors.
        self.NNSaveDir = NNSaveDir
        # size of feature vector
        self.embedSize = embedSize
        # if normalizing == True:
        # output from feature extractor is l2 normlized.
        # default is False.
        self.normalizing = normalizing
        # if useLinLayer == True:
        # linear layer from pretrained model is used.
        # default is False.
        self.useLinLayer = useLinLayer
        # if isInputProb == True:
        # last output from cluster head is softmaxed.
        # default is False
        self.isInputProb = isInputProb

        # which resnet model type will be loaded.
        # resnet18 or resnet50.
        self.modelType = modelType
        if basemodelLoadName in ['imagenet10','imagenet50','imagenet100','imagenet200','tinyimagenet']:
            self.modelType = 'resnet50'
        self.L2NormalEnd = L2NormalEnd

        # size of hidden layer
        self.cDim1 = cDim1
        # number of nearest neighbors
        self.topKNum = topKNum
        # number of nearest neighbors
        self.nnNum= nnNum

        self.configPath = configPath
        # number of cluster class.
        self.clusterNum = clusterNum
        # single fully connected layer or MLP
        self.layerMethod = layerMethod
        # number of cluster head
        # default is 10.
        self.numHeads = numHeads

        self.labelNoiseRatio = labelNoiseRatio
        self.reliableCheckRatio = reliableCheckRatio
        self.reliableCheckNum = reliableCheckNum

        # threshold to for self labeling step.
        self.selfLabelThreshold = selfLabelThreshold
        self.consistencyRatio = consistencyRatio

        self.trnBSize = trnBSize
        self.valBSize = valBSize
        self.numRepeat = numRepeat

        # train batch size for self labeling step.
        self.jointTrnBSize = jointTrnBSize

        self.lossMethod = lossMethod

        # entropy weight for SCAN. default is 5.
        self.entropyWeight = entropyWeight
        # weight for loss (1). default is 1.
        # deprecated.
        self.clusteringWeight = clusteringWeight

        # number of gradient accumulation
        self.accumulNum = accumulNum

        # train cluster head or not.
        # if true: cluster head only is trained.
        self.update_cluster_head_only = update_cluster_head_only

        dataCfg = Config.fromfile(self.configPath)
        if basemodelLoadName == 'cifar10':
            cfgScan = dataCfg.dataConfigs_Cifar10.trans2
            self.baseTransform = dataCfg.dataConfigs_Cifar10.baseTransform
        if basemodelLoadName == 'cifar100':
            cfgScan = dataCfg.dataConfigs_Cifar100.trans2
            self.baseTransform = dataCfg.dataConfigs_Cifar100.baseTransform
        if basemodelLoadName == 'stl10':
            cfgScan = dataCfg.dataConfigs_Stl10.trans2
            self.baseTransform = dataCfg.dataConfigs_Stl10.baseTransform
        if basemodelLoadName in ['imagenet10','imagenet50','imagenet100','imagenet200']:
            cfgScan = dataCfg.dataConfigs_Imagenet.trans2
            self.baseTransform = dataCfg.dataConfigs_Imagenet.baseTransform
        if basemodelLoadName == 'tinyimagenet':
            cfgScan = dataCfg.dataConfigs_tinyImagenet.trans2
            self.baseTransform = dataCfg.dataConfigs_tinyImagenet.baseTransform


        print(cfgScan)
        self.scanTransform = get_train_transformations(cfgScan)

        self.lr = lr
        self.wDecay = wDecay
        self.gpuUse = gpuUse



        self.FeatureExtractorSCAN = callAnyResnet(modelType=self.modelType,
                                                  numClass=self.embedSize,
                                                  normalizing=self.normalizing,
                                                  useLinLayer=self.useLinLayer
                                                  )

        try:
            print(f'loading {self.FESaveLoadDir} {self.FELoadNum}.pt')
            modelStateDict = torch.load(self.FESaveLoadDir +str(self.FELoadNum)+'.pt')
            missing = self.FeatureExtractorSCAN.load_state_dict(modelStateDict)

            print(f'missing : ',set(missing[1]))
            print(f'loading {self.FESaveLoadDir}{self.FELoadNum}.pt complete successfully!~!')
        except:
            if basemodelLoadName not in ['cifar10','cifar100','stl10']:
                print(f'loading base model..')
                loadPretrained4imagenet(baseLoadDir = self.basemodelSaveLoadDir+self.basemodelLoadName,
                                        model=self.FeatureExtractorSCAN)
                print(f'loading base model {self.basemodelSaveLoadDir + self.basemodelLoadName} complete!')
                self.FELoadNum = 0
            else:
                print(f'loading base model..')
                modelStateDict = torch.load(self.basemodelSaveLoadDir + self.basemodelLoadName)
                missing = self.FeatureExtractorSCAN.load_state_dict(modelStateDict,strict=False)
                assert (set(missing[1]) == {
                    'contrastive_head.0.weight', 'contrastive_head.0.bias',
                    'contrastive_head.2.weight', 'contrastive_head.2.bias'}
                        or set(missing[1]) == {
                            'contrastive_head.weight', 'contrastive_head.bias'})
                print(f'loading base model {self.basemodelSaveLoadDir + self.basemodelLoadName} complete!')
                self.FELoadNum = 0

        try:
            self.ClusterHead = myMultiCluster4SCAN(inputDim=self.embedSize,
                                                    dim1=self.cDim1,
                                                    nClusters=self.clusterNum,
                                                    numHead=self.numHeads,
                                                    isOutputProb=self.isInputProb,
                                                    layerMethod= self.layerMethod)

            headLoadedDict= torch.load(self.headSaveLoadDir+str(self.headLoadNum)+'.pt')
            self.ClusterHead.load_state_dict(headLoadedDict)
            print(f'loading saved head : {str(self.headLoadNum)}.pt complete!!!!!!!')


            # self.ClusterHead = DataParallel(self.ClusterHead,device_ids=[0,1,2,3])


        except:
            print('loading saved head failed so start with fresh head')
            self.ClusterHead = myMultiCluster4SCAN(inputDim=self.embedSize,
                                                   dim1=self.cDim1,
                                                   nClusters=self.clusterNum,
                                                   numHead=self.numHeads,
                                                   isOutputProb=self.isInputProb,
                                                   layerMethod=self.layerMethod)
            self.headLoadNum = 0

        if self.gpuUse == True:
            USE_CUDA = torch.cuda.is_available()
            print(USE_CUDA)
            self.device = torch.device('cuda' if USE_CUDA else 'cpu')
            print('학습을 진행하는 기기:', self.device)
        else:
            self.device = torch.device('cpu')
            print('학습을 진행하는 기기:', self.device)


        self.optimizerBackbone = Adam(self.FeatureExtractorSCAN.parameters(),
                                      lr=self.lr,
                                      weight_decay=self.wDecay)

        self.optimizerCHead = Adam(self.ClusterHead.parameters(),
                                   lr=self.lr,
                                   weight_decay=self.wDecay)

        # tmp list for loss (1) when training step a-1
        self.headOnlyConsisLossLst = []
        # list for loss (1) when training step a-1
        self.headOnlyConsisLossLstAvg = []
        # tmp list for entropy loss  when training step a-1
        self.headOnlyEntropyLossLst = []
        # list for entropy loss when training step a-1
        self.headOnlyEntropyLossLstAvg = []
        # tmp list for total loss  when training step a-1
        self.headOnlyTotalLossLst = []
        # list for total loss  when training step a-1
        self.headOnlyTotalLossLstAvg = []

        self.clusterOnlyAccLst = []

        self.jointTrainingLossLst = []
        self.jointTrainingLossLstAvg = []
        self.jointTrainingAccLst = []

        # index of cluster head with minimal loss
        self.minHeadIdx = 0
        self.minHeadIdxLst = []
        self.minHeadIdxJointTraining = 0
        self.minHeadIdxLstJointTraining = []

        self.clusterOnlyClusteringLossDictPerHead = dict()
        for h in range(self.numHeads):
            self.clusterOnlyClusteringLossDictPerHead[f'head_{h}'] = []

        self.clusterOnlyEntropyLossDictPerHead = dict()
        for h in range(self.numHeads):
            self.clusterOnlyEntropyLossDictPerHead[f'head_{h}'] = []

        self.clusterOnlyTotalLossDictPerHead = dict()
        for h in range(self.numHeads):
            self.clusterOnlyTotalLossDictPerHead[f'head_{h}'] = []

        self.jointTrainingLossDictPerHead = dict()
        for h in range(self.numHeads):
            self.jointTrainingLossDictPerHead[f'head_{h}'] = []

    # save nearest neighbor of each data.
    # this code must be executed before training SCAN
    def saveNearestNeighbor(self):

        self.FeatureExtractorSCAN.to(self.device)
        self.ClusterHead.to(self.device)

        dataset4kNN = getCustomizedDataset4SCAN(downDir=self.downDir,
                                                dataType=self.dataType,
                                                transform=self.baseTransform,
                                                baseVer=True)
        dataloader4kNN = DataLoader(dataset4kNN,
                                    batch_size=512,
                                    shuffle=False,
                                    num_workers=2)

        totalFeatures = []
        totalLabels = []
        with torch.set_grad_enabled(False):
            for idx,i in enumerate(dataloader4kNN):
                inputs = i['image'].to(self.device)
                labels = i['label']
                features = self.FeatureExtractorSCAN(inputs).cpu().clone().detach()
                totalFeatures.append(features)
                totalLabels.append(labels)

        # totalFeatures is composed of each feature vector from original image
        totalFeatures = torch.cat(totalFeatures)
        totalLabels = torch.cat(totalLabels)

        index = faiss.IndexFlatIP(totalFeatures.size(1))
        # index = faiss.index_cpu_to_all_gpus(index)
        index.add(totalFeatures.numpy())
        distances, indices = index.search(totalFeatures.numpy(), self.topKNum+1) # Sample itself is included

        # save index of each nearest neighbor of each image into self.NNSaveDir
        np.save(self.NNSaveDir+'NNs.npy',indices)
        print('saving index of nearest neighbors complete')

        self.FeatureExtractorSCAN.to('cpu')
        self.ClusterHead.to('cpu')

    # indicate step B-1.
    # HeadOnly does not mean training head only.
    def trainHeadOnly(self,iterNum):

        self.FeatureExtractorSCAN.to(self.device)
        self.ClusterHead.to(self.device)

        self.FeatureExtractorSCAN.eval()

        # load indices of nerest neighbors of each image.
        indices = np.load(self.NNSaveDir+'NNs.npy')

        # load dataset for scan step b-1 traininig.
        trainDataset = getCustomizedDataset4SCAN(downDir=self.downDir,
                                                 dataType=self.dataType,
                                                 transform=self.scanTransform,
                                                 nnNum= self.nnNum,
                                                 indices=indices,
                                                 toNeighborDataset=True)


        trainDataloader = DataLoader(trainDataset,shuffle=True,batch_size=self.trnBSize,num_workers=2)
        self.ClusterHead.train()

        # repeat training for iterNum.
        # default of iterNum is 1.
        for iter in range(iterNum):
            print(f'{iter}/{iterNum} training Start...')
            totalLossDict,\
            consistencyLossDict,\
            entropLossDict = scanTrain(train_loader= trainDataloader,
                                       featureExtractor = self.FeatureExtractorSCAN,
                                       headNum=self.numHeads,
                                       ClusterHead=self.ClusterHead,
                                       criterion=SCANLoss(entropyWeight=self.entropyWeight,
                                                          isInputProb=self.isInputProb),
                                       accumulNum= self.accumulNum,
                                       optimizer=[self.optimizerBackbone,self.optimizerCHead],
                                       device=self.device,
                                       update_cluster_head_only=self.update_cluster_head_only)

            # append loss to list
            for h in range(self.numHeads):
                self.clusterOnlyTotalLossDictPerHead[f'head_{h}'].append(np.mean(totalLossDict[f'head_{h}']))
                self.clusterOnlyClusteringLossDictPerHead[f'head_{h}'].append(np.mean(consistencyLossDict[f'head_{h}']))
                self.clusterOnlyEntropyLossDictPerHead[f'head_{h}'].append(np.mean(entropLossDict[f'head_{h}']))
            print(f'{iter}/{iterNum} training Complete !!!')

        self.ClusterHead.eval()

        self.FeatureExtractorSCAN.to('cpu')
        self.ClusterHead.to('cpu')

    # validation step of SCAN step B-1.
    # check accruacy when noise ratio is 0.2 for validation
    # noise ratio varies from 0.3 to 0.8 when final test.
    def valHeadOnly(self):

        self.FeatureExtractorSCAN.to(self.device)
        self.ClusterHead.to(self.device)

        self.FeatureExtractorSCAN.eval()
        self.ClusterHead.eval()

        # get index of cluster head with minimal loss
        lst4CheckMinLoss = []
        for h in range(self.numHeads):
            lst4CheckMinLoss.append(np.mean(self.clusterOnlyTotalLossDictPerHead[f'head_{h}']))
        print(f'flushing cluster Loss lst complete')
        # get index of cluster head with minimal loss
        self.minHeadIdx = np.argmin(lst4CheckMinLoss)
        self.minHeadIdxLst.append(self.minHeadIdx)
        trainDataset = getCustomizedDataset4SCAN(downDir=self.downDir,
                                                 dataType=self.dataType,
                                                 transform=self.baseTransform,
                                                 baseVer=True)

        TDataLoader = tqdm(DataLoader(trainDataset,shuffle=True,batch_size=self.trnBSize,num_workers=2))

        clusterPredResult = []
        gtLabelResult = []
        # predict cluster for each inputs
        with torch.set_grad_enabled(False):
            for idx, loadedBatch in enumerate(TDataLoader):

                inputsRaw = loadedBatch['image'].float()
                embededInput = self.FeatureExtractorSCAN(inputsRaw.to(self.device))

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
        print(f'size of gtLabelResult is : {gtLabelResult.size()} and size of clusterpred is : {clusterPredResult.size()}')
        # print(f'clusterPred has size : {clusterPredResult.size()} , gtLabelResult has size : {gtLabelResult.size()}')

        ################################# make noised label with ratio ###############################
        ################################# make noised label with ratio ###############################
        minGtLabel = torch.min(torch.unique(gtLabelResult))
        maxGtLabel = torch.max(torch.unique(gtLabelResult))

        # noisedLabels : var for total labels with noised label
        noisedLabels = []
        # noisedLabels4AccCheck : var for checking accruacy of head, contains noised label only
        noisedLabels4AccCheck = []
        # noiseInserTerm : every interval of this term, noised label is inserted into total labels
        noiseInsertTerm = int(1 / self.labelNoiseRatio)
        print(f'noiseInserTerm is : {noiseInsertTerm}')
        for idx, (eachClusterPred, eachGtLabel) in enumerate(zip(clusterPredResult, gtLabelResult)):
            if idx % noiseInsertTerm == 0:
                noisedLabel = torch.randint(minGtLabel.cpu(),
                                                  maxGtLabel + 1, (1,))
                noisedLabels.append(noisedLabel)
                noisedLabels4AccCheck.append([eachClusterPred.cpu(),
                                              eachGtLabel.cpu(),
                                              noisedLabel])
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


        accCheck = []
        for eachCheckElement in noisedLabels4AccCheck:
            eachPredictedLabel = eachCheckElement[0].item()
            eachGroundTruthLabel = eachCheckElement[1]


            if modelLabelPerCluster[eachPredictedLabel] == eachGroundTruthLabel:
                accCheck.append(1)
            else:
                accCheck.append(0)

        print(f'len of accCheck is : {len(accCheck)}')

        self.clusterOnlyAccLst.append(np.mean(accCheck))
        print(f'validation step end with accuracy : {np.mean(accCheck)}, total OK is : {np.sum(accCheck)} and '
              f'not OK is : {np.sum(np.array(accCheck) == 0)} with '
              f'length of data : {len(accCheck)}')

        self.FeatureExtractorSCAN.to('cpu')
        self.ClusterHead.to('cpu')

    # flush tmp list and save plot of result
    def valHeadOnlyEnd(self):

        self.headOnlyConsisLossLstAvg.append(np.mean(self.clusterOnlyClusteringLossDictPerHead[f'head_{self.minHeadIdx}']))
        self.headOnlyEntropyLossLstAvg.append(
            np.mean(self.clusterOnlyEntropyLossDictPerHead[f'head_{self.minHeadIdx}']))


        for h in range(self.numHeads):
            self.clusterOnlyTotalLossDictPerHead[f'head_{h}'] = []
            self.clusterOnlyClusteringLossDictPerHead[f'head_{h}'] = []
            self.clusterOnlyEntropyLossDictPerHead[f'head_{h}'] = []

        fig = plt.figure(constrained_layout=True)

        ax1 = fig.add_subplot(1, 3, 1)
        ax1.plot(range(len(self.headOnlyConsisLossLstAvg)), self.headOnlyConsisLossLstAvg)
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('clustering loss')


        ax2 = fig.add_subplot(1, 3, 2)
        ax2.plot(range(len(self.headOnlyEntropyLossLstAvg)), self.headOnlyEntropyLossLstAvg)
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('entropy loss')


        ax3 = fig.add_subplot(1, 3, 3)
        ax3.plot(range(len(self.clusterOnlyAccLst)), self.clusterOnlyAccLst)
        ax3.set_xlabel('epoch')
        ax3.set_ylabel('acc')


        plt.savefig(self.plotSaveDir + 'HeadOnlyResult.png', dpi=200)
        print('saving head only plot complete !')
        plt.close()
        plt.clf()
        plt.cla()

        # save list containing indices of cluster head with minimal loss
        with open(self.plotSaveDir+'minLossHeadIdx.pkl','wb') as F:
            pickle.dump(self.minHeadIdxLst,F)
        print('saving head idx of having minimum loss lst')

        # save .pkl file containing validation accruacy of step B-1.
        with open(self.plotSaveDir+'headOnlyAccLst.pkl','wb') as F:
            pickle.dump(self.clusterOnlyAccLst,F)
        print('saving head only acc lst complete')

        # save .pkl file containing loss (1) of step B-1.
        with open(self.plotSaveDir+'headOnlyConsisLossLst.pkl','wb') as F:
            pickle.dump(self.headOnlyConsisLossLstAvg,F)
        print('saving head only acc lst complete')

        # save .pkl file containing entropy loss of step B-1.
        with open(self.plotSaveDir+'headOnlyEntropyLossLst.pkl','wb') as F:
            pickle.dump(self.headOnlyEntropyLossLstAvg,F)
        print('saving head only acc lst complete')


    def executeTrainingHeadOnly(self,iterNum=1):
        # time.sleep(10)
        self.trainHeadOnly(iterNum=iterNum)
        print('training done')
        self.valHeadOnly()
        self.valHeadOnlyEnd()

    # self labeling step, B-2.
    # joint doesn't mean that always feature extracotr and head are trained
    # in the case of imagenet10, cluster head only was trained.
    def trainJointly(self,iterNum=1):

        self.FeatureExtractorSCAN.to(self.device)
        self.ClusterHead.to(self.device)

        trainDataset = getCustomizedDataset4SCAN(downDir=self.downDir,
                                                 dataType=self.dataType,
                                                 transform={'standard':self.baseTransform,
                                                            'augment':self.scanTransform},
                                                 toAgumentedDataset=True)

        trainDataloader = DataLoader(trainDataset,shuffle=True,batch_size=self.jointTrnBSize,num_workers=2)

        self.ClusterHead.train()
        for iter in range(iterNum):
            print(f'{iter}/{iterNum} training Start...')
            totalLossDict = selflabelTrain(train_loader= trainDataloader,
                                           featureExtractor = self.FeatureExtractorSCAN,
                                           headNum=self.numHeads,
                                           ClusterHead=self.ClusterHead,
                                           criterion=selfLabelLoss(selfLabelThreshold=self.selfLabelThreshold,
                                                                   isInputProb=self.isInputProb),
                                           optimizer=[self.optimizerBackbone,self.optimizerCHead],
                                           device=self.device,
                                           accumulNum=self.accumulNum,
                                           update_cluster_head_only=self.update_cluster_head_only)

            # append losses to list
            for h in range(self.numHeads):
                self.jointTrainingLossDictPerHead[f'head_{h}'].append(np.mean(totalLossDict[f'head_{h}']))

            print(f'{iter}/{iterNum} training Complete !!!')

        self.ClusterHead.eval()

        self.FeatureExtractorSCAN.to('cpu')
        self.ClusterHead.to('cpu')

    # validation step of stepB-2.
    # accuracy when noise ratio is 0.2 is calculated.
    # noise ratio varies from 0.3 to 0.8 when final test.
    def jointVal(self):

        self.FeatureExtractorSCAN.to(self.device)
        self.ClusterHead.to(self.device)

        self.FeatureExtractorSCAN.eval()
        self.ClusterHead.eval()

        # find indices of head with minimal loss
        lst4CheckMinLoss = []
        for h in range(self.numHeads):
            lst4CheckMinLoss.append(np.mean(self.jointTrainingLossDictPerHead[f'head_{h}']))
        print(f'flushing cluster Loss lst complete')
        # find indices of head with minimal loss
        self.minHeadIdxJointTraining = np.argmin(lst4CheckMinLoss)
        self.minHeadIdxLstJointTraining.append(self.minHeadIdxJointTraining)
        trainDataset = getCustomizedDataset4SCAN(downDir=self.downDir,
                                                 dataType=self.dataType,
                                                 transform=self.baseTransform,
                                                 baseVer=True)

        TDataLoader = tqdm(DataLoader(trainDataset, shuffle=True, batch_size=self.trnBSize, num_workers=2))

        clusterPredResult = []
        gtLabelResult = []
        # predict cluster for each inputs
        with torch.set_grad_enabled(False):
            for idx, loadedBatch in enumerate(TDataLoader):
                inputsRaw = loadedBatch['image'].float()
                embededInput = self.FeatureExtractorSCAN(inputsRaw.to(self.device))


                # clusterProb = self.ClusterHead.forwardWithMinLossHead(embededInput,
                #                                                       headIdxWithMinLoss=self.minHeadIdxJointTraining)

                clusterProb = self.ClusterHead.forward(embededInput,headIdxWithMinLoss=self.minHeadIdxJointTraining)

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
        print(f'clusterPred has size : {clusterPredResult.size()} , gtLabelResult has size : {gtLabelResult.size()}')

        ################################# make noised label with ratio ###############################
        ################################# make noised label with ratio ###############################
        minGtLabel = torch.min(torch.unique(gtLabelResult))
        maxGtLabel = torch.max(torch.unique(gtLabelResult))

        # noisedLabels : var for total labels with noised label
        noisedLabels = []
        # noisedLabels4AccCheck : var for checking accruacy of head, contains noised label only
        noisedLabels4AccCheck = []
        # noiseInserTerm : every interval of this term, noised label is inserted into total labels
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

            if modelLabelPerCluster[eachPredictedLabel] == eachGroundTruthLabel:
                accCheck.append(1)
            else:
                accCheck.append(0)

        print(f'len of accCheck is : {len(accCheck)}')
        self.jointTrainingAccLst.append(np.mean(accCheck))
        print(f'validation step end with accuracy : {np.mean(accCheck)}, total OK is : {np.sum(accCheck)} and '
              f'not OK is : {np.sum(np.array(accCheck) == 0)} with '
              f'length of data : {len(accCheck)}')

        self.FeatureExtractorSCAN.to('cpu')
        self.ClusterHead.to('cpu')

    # flush tmp list and save result and plot
    def jointValEnd(self):

        self.jointTrainingLossLstAvg.append(
            np.mean(self.jointTrainingLossDictPerHead[f'head_{self.minHeadIdxJointTraining}']))

        for h in range(self.numHeads):
            self.jointTrainingLossDictPerHead[f'head_{h}'] = []

        fig = plt.figure(constrained_layout=True)

        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(range(len(self.jointTrainingLossLstAvg)), self.jointTrainingLossLstAvg)
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('self labeling loss')


        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(range(len(self.jointTrainingAccLst)), self.jointTrainingAccLst)
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('self labeling acc')


        plt.savefig(self.plotSaveDir + 'selfLabelingResult.png', dpi=200)
        print('saving self labeling plot complete !')
        plt.close()
        plt.clf()
        plt.cla()

        # save indices of head with minimal losses
        with open(self.plotSaveDir + 'minLossHeadIdxJointTraining.pkl', 'wb') as F:
            pickle.dump(self.minHeadIdxLstJointTraining, F)
        print('saving head idx of having minimum loss lst complete')

        # save accuracy of validations when step B-2
        with open(self.plotSaveDir + 'jointTrnAccLst.pkl', 'wb') as F:
            pickle.dump(self.jointTrainingAccLst, F)

        # save loss when step B-2
        with open(self.plotSaveDir + 'jointTrnLossLst.pkl', 'wb') as F:
            pickle.dump(self.jointTrainingLossLstAvg, F)

        print('saving joint trainig acc lst complete ')

    def executeJointTraining(self,iterNum=1):
        # time.sleep(10)
        self.trainJointly(iterNum=iterNum)
        print('training done')
        self.jointVal()
        self.jointValEnd()

    def saveHead(self,iteredNum):
        torch.save(self.ClusterHead.state_dict(),self.headSaveLoadDir+str(iteredNum+self.headLoadNum)+'.pt')
        print(f'saving head complete!!!')
        print(f'saving head complete!!!')
        print(f'saving head complete!!!')

    def saveFeatureExtractor(self,iteredNum):
        torch.save(self.FeatureExtractorSCAN.state_dict(), self.FESaveLoadDir + str(iteredNum + self.FELoadNum) + '.pt')
        print(f'saving head complete!!!')
        print(f'saving head complete!!!')
        print(f'saving head complete!!!')

    # check accuracy per confidence
    def checkConfidence(self):

        self.FeatureExtractorSCAN.to(self.device)
        self.ClusterHead.to(self.device)


        self.FeatureExtractorSCAN.eval()
        self.ClusterHead.eval()

        # get index of head with minimal loss
        self.minHeadIdx = getMinHeadIdx(self.plotSaveDir)
        print(f'self.minHeadIdx is : {self.minHeadIdx}')
        trainDataset = getCustomizedDataset4SCAN(downDir=self.downDir,
                                                 dataType=self.dataType,
                                                 transform=self.baseTransform,
                                                 baseVer=True)
        TDataLoader = tqdm(DataLoader(trainDataset,shuffle=True,batch_size=self.trnBSize,num_workers=2))

        clusterPredResult = []
        clusterPredValueResult = []
        gtLabelResult = []
        # predict cluster for each inputs
        with torch.set_grad_enabled(False):
            for idx, loadedBatch in enumerate(TDataLoader):

                inputsRaw = loadedBatch['image'].float()
                embededInput = self.FeatureExtractorSCAN(inputsRaw.to(self.device))
                clusterProb = self.ClusterHead.forwardWithMinLossHead(embededInput, headIdxWithMinLoss=self.minHeadIdx).cpu()

                clusterPred = torch.argmax(clusterProb, dim=1)
                clusterPredValue = torch.max(F.softmax(clusterProb,dim=1),dim=1).values.cpu()

                clusterPredResult.append(clusterPred)
                clusterPredValueResult.append(clusterPredValue)
                gtLabelResult.append(loadedBatch['label'])


        # result of prediction for each inputs
        clusterPredResult = torch.cat(clusterPredResult)
        for eachClusterUnique in torch.unique(clusterPredResult):
            print(
                f' {torch.count_nonzero(clusterPredResult == eachClusterUnique)} allocated for cluster :{eachClusterUnique}'
                f'when validating')

        clusterPredValueResult = torch.cat(clusterPredValueResult)

        # ground truth label for each inputs
        gtLabelResult = torch.cat(gtLabelResult).unsqueeze(1)
        print(f'size of gtLabelResult is : {gtLabelResult.size()}')
        print(f'clusterPred has size : {clusterPredResult.size()} , gtLabelResult has size : {gtLabelResult.size()}')

        ################################# make noised label with ratio ###############################
        ################################# make noised label with ratio ###############################
        minGtLabel = torch.min(torch.unique(gtLabelResult))
        maxGtLabel = torch.max(torch.unique(gtLabelResult))

        # noisedLabels : var for total labels with noised label
        noisedLabels = []
        # noisedLabels4AccCheck : var for checking accruacy of head, contains noised label only
        noisedLabels4AccCheck = []
        # noiseInserTerm : every interval of this term, noised label is inserted into total labels
        noiseInsertTerm = int(1 / self.labelNoiseRatio)
        print(f'noiseInserTerm is : {noiseInsertTerm}')
        for idx, (eachClusterPred, eachGtLabel, eachClusterPredValue) in enumerate(zip(clusterPredResult,
                                                                                      gtLabelResult,
                                                                                      clusterPredValueResult)):
            if idx % noiseInsertTerm == 0:
                noisedLabels.append(torch.randint(minGtLabel.cpu(),
                                                  maxGtLabel + 1, (1,)))
                noisedLabels4AccCheck.append([eachClusterPred.cpu(),
                                              eachGtLabel.cpu(),
                                              torch.randint(minGtLabel, maxGtLabel + 1, (1,)),
                                              eachClusterPredValue])
            else:
                noisedLabels.append(eachGtLabel)
                noisedLabels4AccCheck.append([eachClusterPred.cpu(),
                                              eachGtLabel.cpu(),
                                              torch.randint(minGtLabel, maxGtLabel + 1, (1,)),
                                              eachClusterPredValue])
        noisedLabels = torch.cat(noisedLabels)
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


        accCheck = dict()
        for eachCheckElement in noisedLabels4AccCheck:
            eachPredictedLabel = eachCheckElement[0].item()
            eachGroundTruthLabel = eachCheckElement[1]
            eachPredictValue = eachCheckElement[3]

            if modelLabelPerCluster[eachPredictedLabel] == eachGroundTruthLabel:
                accResult = 1
            else:
                accResult = 0

            accCheck[eachPredictValue] = accResult

        finalConf, finalAcc,finalAllocNum = getAccPerConfLst(accCheck,10,minConf=0.95)

        # save plot
        plt.bar(finalConf,finalAcc)
        plt.xlabel('Probability Range')
        plt.xticks(rotation=30)
        plt.ylabel('Acc')
        plt.savefig(self.plotSaveDir+'accPerConf.png',dpi=200,bbox_inches='tight')
        plt.close()
        plt.cla()
        plt.clf()

        plt.bar(finalConf, finalAllocNum)
        plt.xlabel('Probability Range')
        plt.xticks(rotation=30)
        plt.ylabel('Allocated Num')
        plt.savefig(self.plotSaveDir + 'AllocPerConf.png', dpi=300,bbox_inches='tight')
        plt.close()
        plt.cla()
        plt.clf()

        self.FeatureExtractorSCAN.to('cpu')
        self.ClusterHead.to('cpu')


    # save data and cluster with high confidence
    def saveFiltered(self):

        self.FeatureExtractorSCAN.to(self.device)
        self.ClusterHead.to(self.device)

        self.FeatureExtractorSCAN.eval()
        self.ClusterHead.eval()

        self.minHeadIdx = getMinHeadIdx(self.plotSaveDir)
        print(f'self.minHeadIdx is : {self.minHeadIdx}')

        trainDataset = getCustomizedDataset4SCAN(downDir=self.downDir,
                                                 dataType=self.dataType,
                                                 transform=self.baseTransform,
                                                 baseVer=True)

        TDataLoader = tqdm(DataLoader(trainDataset, shuffle=True, batch_size=self.trnBSize, num_workers=2))
        filteredInputLst = []
        filteredClusterLst =  []
        with torch.set_grad_enabled(False):
            for idx, loadedBatch in enumerate(TDataLoader):
                inputsRaw = loadedBatch['image'].float()
                
                inputsIndices = loadedBatch['meta']['index']

                embededInput = self.FeatureExtractorSCAN(inputsRaw.to(self.device))
                clusterProb = self.ClusterHead.forwardWithMinLossHead(embededInput,
                                                                      headIdxWithMinLoss=self.minHeadIdx).cpu()
                clusterPred = torch.argmax(clusterProb, dim=1)
                clusterPredValue = torch.max(torch.nn.functional.softmax(clusterProb,dim=1), dim=1).values.cpu()

                confMask = clusterPredValue >= self.selfLabelThreshold

                filteredInputLst.append(inputsIndices[confMask])
                filteredClusterLst.append(clusterPred[confMask])
                # filteredClusterLst.append(Pseudo2Label(modelLabelPerCluster, clusterPred)[confMask])

        filteredInputLst = torch.cat(filteredInputLst)
        filteredClusterLst = torch.cat(filteredClusterLst)

        print(f'size of filtered input is : {filteredInputLst.size()}')
        print(f'size of filtered pseudo label is : {filteredClusterLst.size()}')

        finalDict = {
            'inputIndices' : filteredInputLst,
            'clusters' : filteredClusterLst
        }

        with open(self.plotSaveDir+f'filteredData_{self.selfLabelThreshold}.pkl','wb') as F:
            pickle.dump(finalDict,F)

        print('saving confident data indices and cluster complete ')

        self.FeatureExtractorSCAN.to('cpu')
        self.ClusterHead.to('cpu')

    # save data which is noised
    # this code is for further study
    def saveNoiseDataIndices(self,theNoise):

        self.FeatureExtractorSCAN.to(self.device)
        self.ClusterHead.to(self.device)

        self.FeatureExtractorSCAN.eval()
        self.ClusterHead.eval()

        self.minHeadIdx = getMinHeadIdx(self.plotSaveDir)
        print(f'self.minHeadIdx is : {self.minHeadIdx}')
        trainDataset = getCustomizedDataset4SCAN(downDir=self.downDir,
                                                 dataType=self.dataType,
                                                 transform=self.baseTransform,
                                                 baseVer=True)
        TDataLoader = tqdm(DataLoader(trainDataset, shuffle=True, batch_size=self.trnBSize, num_workers=2))

        clusterPredResult = []
        indicesLst = []
        clusterPredValueResult = []
        gtLabelResult = []
        # predict cluster for each inputs
        with torch.set_grad_enabled(False):
            for idx, loadedBatch in enumerate(TDataLoader):
                inputsRaw = loadedBatch['image'].float()

                inputsIndices = loadedBatch['meta']['index']

                embededInput = self.FeatureExtractorSCAN(inputsRaw.to(self.device))
                clusterProb = self.ClusterHead.forwardWithMinLossHead(embededInput,
                                                                      headIdxWithMinLoss=self.minHeadIdx).cpu()

                clusterPred = torch.argmax(clusterProb, dim=1)
                clusterPredValue = torch.max(clusterProb, dim=1).values.cpu()

                clusterPredResult.append(clusterPred)
                clusterPredValueResult.append(clusterPredValue)
                gtLabelResult.append(loadedBatch['label'])
                indicesLst.append(inputsIndices)

        indicesLst = torch.cat(indicesLst)
        # result of prediction for each inputs
        clusterPredResult = torch.cat(clusterPredResult)
        for eachClusterUnique in torch.unique(clusterPredResult):
            print(
                f' {torch.count_nonzero(clusterPredResult == eachClusterUnique)} allocated for cluster :{eachClusterUnique}'
                f'when validating')

        clusterPredValueResult = torch.cat(clusterPredValueResult)

        # ground truth label for each inputs
        gtLabelResult = torch.cat(gtLabelResult).unsqueeze(1)
        # for i in range(100):
        #     print(f'size of gtLabelResult is : {gtLabelResult.size()} while clusterPredResult size is : {clusterPredResult.size()}')
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
        noiseInsertTerm = int(1 / theNoise)
        print(f'noiseInserTerm is : {noiseInsertTerm}')

        noiseOrNot = dict()
        for idx, (eachClusterPred, eachGtLabel, eachClusterPredValue,eachIndex) in enumerate(zip(clusterPredResult,
                                                                                                 gtLabelResult,
                                                                                                 clusterPredValueResult,
                                                                                                 indicesLst)):
            if idx % noiseInsertTerm == 0:

                noisedLabel = torch.randint(minGtLabel.cpu(),
                                                  maxGtLabel + 1, (1,))
                noisedLabels.append(noisedLabel)
                noisedLabels4AccCheck.append([eachIndex.item(),
                                              eachGtLabel.cpu().clone().detach().item()
                                              ])
                noiseOrNot[eachIndex] = True
            else:
                noisedLabels.append(eachGtLabel)
                noiseOrNot[eachIndex] = False

        noisedLabels = torch.cat(noisedLabels)

        with open(self.plotSaveDir+f'noisedDataOnly_{str(theNoise)}.csv', 'w') as F:
            wr = csv.writer(F)
            wr.writerows(noisedLabels4AccCheck)

        noisedDataDict = {
            'noiseOrNot' : noiseOrNot
        }

        # with open(self.plotSaveDir+f'noisedDataOnly_{str(theNoise)}.pkl', 'wb') as F:
        #     pickle.dump(noisedDataDict,F)
        print('saving noised Data only complete')
        ################################# make noised label with ratio ###############################
        ################################# make noised label with ratio ###############################

        # dict containing which labels is mode in each cluster
        modelLabelPerCluster = dict()
        for eachCluster in range(self.clusterNum):
            sameClusterIdx = clusterPredResult == eachCluster
            modeLabel = torch.mode(noisedLabels[sameClusterIdx]).values
            try:
                modelLabelPerCluster[eachCluster] = modeLabel
            except:
                modelLabelPerCluster[eachCluster] = 0

        with open(self.plotSaveDir+f'cluster2label_{str(theNoise)}.pkl', 'wb') as F:
            pickle.dump(modelLabelPerCluster,F)
        print('saving cluster 2 label complete')


    # load model for training with high confident data only
    # this code is for further study.
    def loadModel4filtered(self,nClass):

        self.FeatureExtractorSCAN.to('cpu')
        self.ClusterHead.to('cpu')

        self.nClass =nClass

        self.FeatureExtractor4FTed = callAnyResnet(modelType=self.modelType,
                                                      numClass=self.embedSize,
                                                      normalizing=False,
                                                      useLinLayer=False,
                                                      )


        print(f'loading {self.FTedFESaveLoadDir} {self.FTedFELoadNum}.pt')
        try:
            modelStateDict = torch.load(self.FTedFESaveLoadDir + str(self.FTedFELoadNum) + '.pt')
            missing = self.FeatureExtractor4FTed.load_state_dict(modelStateDict)
            print(f'missing : ', set(missing[1]))
            # assert (set(missing[1]) == {
            #     'contrastive_head.0.weight', 'contrastive_head.0.bias',
            #     'contrastive_head.2.weight', 'contrastive_head.2.bias'}
            #         or set(missing[1]) == {
            #             'contrastive_head.weight', 'contrastive_head.bias'})
            print(f'loading {self.FTedFESaveLoadDir}{self.FTedFELoadNum}.pt complete successfully!~!')
        except:
            print('loading feature extractor failed')
            self.FTedFELoadNum = 0

        self.ClusterHeadFTed = myPredictorHead(inputDim=self.embedSize,
                                           dim1=self.cDim1,
                                           nClass=self.nClass)
        try:
            headLoadedDict = torch.load(self.FTedheadSaveLoadDir + str(self.FTedheadLoadNum) + '.pt')
            self.ClusterHeadFTed.load_state_dict(headLoadedDict)
            print(f'loading saved head : {str(self.FTedheadLoadNum)}.pt complete!!!!!!!')
        except:
            print('loading head failed')
            self.FTedheadLoadNum = 0

        self.optimizerBackboneFTed = Adam(self.FeatureExtractor4FTed.parameters(),
                                      lr=self.lr,
                                      weight_decay=self.wDecay)

        self.optimizerCHeadFTed = Adam(self.ClusterHeadFTed.parameters(),
                                   lr=self.lr,
                                   weight_decay=self.wDecay)

        self.ftedTrainLossLst = []
        self.ftedTrainAccLst = []
        self.ftedValAcc4TotalLst = []
        self.ftedValAcc4NoiseOnlyLst = []

    # train model with high confident data only
    # this code is for further study
    def trainFilteredDataNaiveVer(self,theNoise):

        self.FeatureExtractor4FTed.to(self.device)
        self.ClusterHeadFTed.to(self.device)

        trainDataset = filteredDatasetNaive4SCAN(downDir=self.downDir,
                                                 savedIndicesDir = self.plotSaveDir,
                                                 dataType=self.dataType,
                                                 noiseRatio = theNoise,
                                                 threshold = self.selfLabelThreshold,
                                                 transform= self.baseTransform
                                                 )

        trainDataloader = DataLoader(trainDataset, shuffle=True, batch_size=self.jointTrnBSize, num_workers=2)

        self.ClusterHeadFTed.train()

        print(f'FTed training Start...')
        totalLossLst,totalAccLst = trainWithFiltered(train_loader=trainDataloader,
                                                     featureExtractor=self.FeatureExtractor4FTed,
                                                     ClusterHead=self.ClusterHeadFTed,
                                                     criterion=filteredDataLoss(),
                                                     optimizer=[self.optimizerBackboneFTed, self.optimizerCHeadFTed],
                                                     device=self.device,
                                                     accumulNum=self.accumulNum
                                                     )

        self.ftedTrainLossLst.append(np.mean(totalLossLst))
        self.ftedTrainAccLst.append(np.mean(totalAccLst))

        print(f'FTed training Complete !!!')

        self.ClusterHeadFTed.eval()

        self.FeatureExtractor4FTed.to('cpu')
        self.ClusterHeadFTed.to('cpu')

    # validate model with high confident data only
    # this code is for further study
    def valFilteredDataNaiveVer(self,theNoise):

        self.FeatureExtractor4FTed.to(self.device)
        self.ClusterHeadFTed.to(self.device)

        self.FeatureExtractor4FTed.eval()
        self.ClusterHeadFTed.eval()

        trainDataset = getCustomizedDataset4SCAN(downDir=self.downDir,
                                                 dataType=self.dataType,
                                                 transform=self.baseTransform,
                                                 baseVer=True)

        TDataLoader = tqdm(DataLoader(trainDataset, shuffle=True, batch_size=self.trnBSize, num_workers=2))

        labelPredResult = []
        gtLabelResult = []
        # predict cluster for each inputs
        with torch.set_grad_enabled(False):
            for idx, loadedBatch in enumerate(TDataLoader):
                inputsRaw = loadedBatch['image'].float()
                embededInput = self.FeatureExtractor4FTed(inputsRaw.to(self.device))
                labelProb = self.ClusterHeadFTed(embededInput)
                labelPred = torch.argmax(labelProb, dim=1).cpu()

                # print(f'clusterPred hase unique ele : {torch.unique(clusterPred)}')
                labelPredResult.append(labelPred)
                gtLabelResult.append(loadedBatch['label'])
                # print('bath size of label is :', loadedBatch['label'].size())

        # result of prediction for each inputs
        labelPredResult = torch.cat(labelPredResult)
        # for eachLabelUnique in torch.unique(labelPredResult):
        #     print(
        #         f' {torch.count_nonzero(labelPredResult == eachLabelUnique)} '
        #         f'allocated for cluster :{eachLabelUnique}'
        #         f'when validating')

        # ground truth label for each inputs
        gtLabelResult = torch.cat(gtLabelResult)
        print(f'size of gtLabelResult is : {gtLabelResult.size()} and preResult is : {labelPredResult.size()}')
        assert gtLabelResult.size() == labelPredResult.size()

        # print(f'clusterPred has size : {clusterPredResult.size()} , gtLabelResult has size : {gtLabelResult.size()}')

        acc4Total = torch.mean((gtLabelResult==labelPredResult).float())

        acc4noiseOnly = []
        trainDataset = noisedOnlyDatasetNaive4SCAN(downDir=self.downDir,
                                                   savedIndicesDir = self.plotSaveDir,
                                                   dataType=self.dataType,
                                                   noiseRatio = theNoise,
                                                   transform= self.baseTransform
                                                   )

        TDataLoader = tqdm(DataLoader(trainDataset, shuffle=True, batch_size=self.trnBSize, num_workers=2))

        labelPredResult = []
        gtLabelResult = []
        # predict cluster for each inputs
        with torch.set_grad_enabled(False):
            for idx, loadedBatch in enumerate(TDataLoader):
                inputsRaw = loadedBatch['image'].float()
                embededInput = self.FeatureExtractor4FTed(inputsRaw.to(self.device))
                labelProb = self.ClusterHeadFTed(embededInput)
                labelPred = torch.argmax(labelProb, dim=1).cpu()

                # print(f'clusterPred hase unique ele : {torch.unique(clusterPred)}')
                labelPredResult.append(labelPred)
                gtLabelResult.append(loadedBatch['label'])
                # print('bath size of label is :', loadedBatch['label'].size())

        # result of prediction for each inputs
        labelPredResult = torch.cat(labelPredResult)
        # ground truth label for each inputs
        gtLabelResult = torch.cat(gtLabelResult)

        assert labelPredResult.size() == gtLabelResult.size()

        acc4noiseOnly = torch.mean((gtLabelResult==labelPredResult).float())

        print(f'acc total is :{acc4Total} and acc noise only is : {acc4noiseOnly}')

        self.ftedValAcc4TotalLst.append(acc4Total)
        self.ftedValAcc4NoiseOnlyLst.append(acc4noiseOnly)

        self.FeatureExtractor4FTed.to('cpu')
        self.ClusterHeadFTed.to('cpu')



    def valFilteredDataNaiveVerEnd(self,theNoise):

        fig = plt.figure(constrained_layout=True)

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(range(len(self.ftedTrainLossLst)), self.ftedTrainLossLst)
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('filtered train loss')
        # ax1.set_title('Head Only Train Loss clustering')

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(range(len(self.ftedTrainAccLst)), self.ftedTrainAccLst)
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('filtered train acc')
        # ax3.set_title(f'val acc , noise ratio : {self.labelNoiseRatio}')

        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(range(len(self.ftedValAcc4TotalLst)), self.ftedValAcc4TotalLst)
        ax3.set_xlabel('epoch')
        ax3.set_ylabel('filtered total acc')
        # ax1.set_title('Head Only Train Loss clustering')

        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(range(len(self.ftedValAcc4NoiseOnlyLst)), self.ftedValAcc4NoiseOnlyLst)
        ax4.set_xlabel('epoch')
        ax4.set_ylabel('filtered noiseonly acc')
        # ax3.set_title(f'val acc , noise ratio : {self.labelNoiseRatio}')

        plt.savefig(self.plotSaveDir + f'ftedTrainingResult_{self.selfLabelThreshold}_{theNoise}.png', dpi=200)
        print('saving filtered training plot complete !')
        plt.close()
        plt.clf()
        plt.cla()

        with open(self.plotSaveDir+f'ftedTrainingResultLst_{self.selfLabelThreshold}_{theNoise}.csv','w') as F:
            wr = csv.writer(F)
            wr.writerows([self.ftedTrainLossLst,
                          self.ftedTrainAccLst,
                          self.ftedValAcc4TotalLst,
                          self.ftedValAcc4NoiseOnlyLst])


    def executeFTedTraining(self,theNoise):

        self.trainFilteredDataNaiveVer(theNoise=theNoise)
        self.valFilteredDataNaiveVer(theNoise=theNoise)
        self.valFilteredDataNaiveVerEnd(theNoise=theNoise)

    def saveFTedModels(self,iteredNum):

        torch.save(self.FeatureExtractor4FTed.state_dict(), self.FTedFESaveLoadDir + str(iteredNum + self.FTedFELoadNum) + '.pt')
        torch.save(self.ClusterHeadFTed.state_dict(),
                   self.FTedheadSaveLoadDir + str(iteredNum + self.FTedheadLoadNum) + '.pt')

        print(f'saving FTed Models complete!!!')
        print(f'saving FTed Models complete!!!')
        print(f'saving FTed Models complete!!!')


    # calculate accuracy per every normal data ratio in list
    def checkAccPerNoise(self,normalRatioLst):

        # list contains scope (1) : of noised data only
        accLstNoiseOnly = []
        # list contains scope (2) : total data
        accLstTotal = []

        self.FeatureExtractorSCAN.to(self.device)
        self.ClusterHead.to(self.device)

        self.FeatureExtractorSCAN.eval()
        self.ClusterHead.eval()

        for eachNormalRatio in normalRatioLst:

            eachNoiseRatio = 1-eachNormalRatio

            # assumes that training and test are executed in continuous manners.
            # in the case of doing test seperately, list containing indices of minimal loss head
            # must be loaded in advance.
            lst4CheckMinLoss = []
            for h in range(self.numHeads):
                lst4CheckMinLoss.append(np.mean(self.jointTrainingLossDictPerHead[f'head_{h}']))
            print(f'flushing cluster Loss lst complete')

            self.minHeadIdxJointTraining = np.argmin(lst4CheckMinLoss)
            self.minHeadIdxLstJointTraining.append(self.minHeadIdxJointTraining)
            trainDataset = getCustomizedDataset4SCAN(downDir=self.downDir,
                                                     dataType=self.dataType,
                                                     transform=self.baseTransform,
                                                     baseVer=True)

            TDataLoader = tqdm(DataLoader(trainDataset, shuffle=True, batch_size=self.trnBSize, num_workers=2))

            clusterPredResult = []
            gtLabelResult = []
            # predict cluster for each inputs
            with torch.set_grad_enabled(False):
                for idx, loadedBatch in enumerate(TDataLoader):
                    inputsRaw = loadedBatch['image'].float()
                    embededInput = self.FeatureExtractorSCAN(inputsRaw.to(self.device))

                    # clusterProb = self.ClusterHead.forwardWithMinLossHead(embededInput,
                    #                                                       headIdxWithMinLoss=self.minHeadIdxJointTraining)

                    clusterProb = self.ClusterHead.forward(embededInput, headIdxWithMinLoss=self.minHeadIdxJointTraining)

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
            print(f'clusterPred has size : {clusterPredResult.size()} , gtLabelResult has size : {gtLabelResult.size()}')

            ################################# make noised label with ratio ###############################
            ################################# make noised label with ratio ###############################
            minGtLabel = torch.min(torch.unique(gtLabelResult))
            maxGtLabel = torch.max(torch.unique(gtLabelResult))

            # noisedLabels : var for total labels with noised label
            noisedLabels = []
            # noisedLabels4AccCheck : var for checking accruacy of head, contains noised label only
            noisedLabels4AccCheck = []
            # noiseInserTerm : every interval of this term, noised label is inserted into total labels
            noiseInsertTerm = int(1 / eachNoiseRatio)
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

                if modelLabelPerCluster[eachPredictedLabel] == eachGroundTruthLabel:
                    accCheck.append(1)
                else:
                    accCheck.append(0)

            accCheckTotal = []
            for idx, (eachClusterPred, eachGtLabel) in enumerate(zip(clusterPredResult, gtLabelResult)):
                if modelLabelPerCluster[eachClusterPred.item()] == eachGtLabel:
                    accCheckTotal.append(1)
                else:
                    accCheckTotal.append(0)

            accLstNoiseOnly.append(np.mean(accCheck))
            accLstTotal.append(np.mean(accCheckTotal))




            print(f'len of accCheck is : {len(accCheck)}')

            print(f'validation step end with accuracy for noise only: {np.mean(accCheck)}, total OK is : {np.sum(accCheck)} and '
                  f'not OK is : {np.sum(np.array(accCheck) == 0)} with '
                  f'length of data : {len(accCheck)}')
            print(f'validation step end with accuracy for total : {np.mean(accCheckTotal)}, total OK is : {np.sum(accCheckTotal)} and '
                  f'not OK is : {np.sum(np.array(accCheckTotal) == 0)} with '
                  f'length of data : {len(accCheckTotal)}')

        # plt.plot(normalRatioLst,accLstNoiseOnly)
        # plt.xlabel('Nornal Data Ratio')
        # plt.ylabel('Accuracy')
        # plt.title('Acc for Noised Data Only')
        # plt.savefig(self.plotSaveDir+'accPerNoise_noiseOnly.png',dpi=200)
        # # plt.close()
        # # plt.cla()
        # # plt.clf()
        #
        # print('saving noise only acc per noise png complete')
        #
        # plt.plot(normalRatioLst, accLstTotal)
        # plt.xlabel('Normal Data Ratio')
        # plt.ylabel('Accuracy')
        # plt.title('Acc for Total Data')
        # plt.savefig(self.plotSaveDir + 'accPerNoise_Total.png', dpi=200)
        # plt.close()
        # plt.cla()
        # plt.clf()

        # print('saving Total acc per noise png complete')

        fig = plt.figure(constrained_layout=True)

        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(normalRatioLst, accLstNoiseOnly)
        ax1.set_xlabel('Normal Data Ratio')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Acc for Noised Data Only')

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(normalRatioLst, accLstTotal)
        ax2.set_xlabel('Normal Data Ratio')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Acc for Total Data')

        plt.savefig(self.plotSaveDir + 'AccsPerNoiseRatioResult.png', dpi=200)
        print('saving Total acc per noise png complete')
        plt.close()
        plt.clf()
        plt.cla()


        accDict = {
            'accNoiseOnly' : accLstNoiseOnly,
            'accTotal' : accLstTotal
        }

        # save results into pkl file
        with open(self.plotSaveDir+'accPerNoiseRatioDict.pkl','wb') as F:
            pickle.dump(accDict,F)

        # save results into csv file
        with open(self.plotSaveDir+'accPerNoiseRatioLst.csv','w') as F:
            wr = csv.writer(F)
            wr.writerow(accLstNoiseOnly)
            wr.writerow(accLstTotal)

        self.FeatureExtractorSCAN.to('cpu')
        self.ClusterHead.to('cpu')
























































