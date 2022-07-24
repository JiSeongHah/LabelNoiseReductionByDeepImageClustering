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
from SCAN_DATASETS import getCustomizedDataset4SCAN,filteredDatasetNaive4SCAN
from SCAN_trainingProcedure import scanTrain,selflabelTrain
from SCAN_losses import SCANLoss,selfLabelLoss,filteredDataLoss
from SCAN_usefulUtils import getMinHeadIdx,getAccPerConfLst,loadPretrained4imagenet,Pseudo2Label
import faiss


class doSCAN(nn.Module):
    def __init__(self,
                 basemodelSaveLoadDir,
                 basemodelLoadName,
                 headSaveLoadDir,
                 FESaveLoadDir,
                 FELoadNum,
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
                 topKNum = 20,
                 selfLabelThreshold=0.99,
                 downDir='/home/a286/hjs_dir1/mySCAN0/',
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


        self.headSaveLoadDir = headSaveLoadDir
        self.FESaveLoadDir = FESaveLoadDir
        self.FELoadNum = FELoadNum
        self.headLoadNum = headLoadNum
        self.plotSaveDir = plotSaveDir
        createDirectory(self.plotSaveDir)
        self.downDir = downDir
        self.NNSaveDir = NNSaveDir
        self.embedSize = embedSize
        self.normalizing = normalizing
        self.useLinLayer = useLinLayer
        self.isInputProb = isInputProb

        self.modelType = modelType
        if basemodelLoadName in ['imagenet10','imagenet50','imagenet100','imagenet200','tinyimagenet']:
            self.modelType = 'resnet50'
        self.L2NormalEnd = L2NormalEnd
        self.cDim1 = cDim1
        self.topKNum = topKNum
        self.nnNum= nnNum
        self.configPath = configPath
        self.clusterNum = clusterNum
        self.layerMethod = layerMethod
        self.numHeads = numHeads
        self.labelNoiseRatio = labelNoiseRatio
        self.reliableCheckRatio = reliableCheckRatio
        self.reliableCheckNum = reliableCheckNum
        self.selfLabelThreshold = selfLabelThreshold
        self.consistencyRatio = consistencyRatio
        self.trnBSize = trnBSize
        self.valBSize = valBSize
        self.numRepeat = numRepeat
        self.jointTrnBSize = jointTrnBSize
        self.lossMethod = lossMethod
        self.entropyWeight = entropyWeight
        self.clusteringWeight = clusteringWeight
        self.accumulNum = accumulNum
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

        if self.gpuUse == True:
            USE_CUDA = torch.cuda.is_available()
            print(USE_CUDA)
            self.device = torch.device('cuda' if USE_CUDA else 'cpu')
            print('학습을 진행하는 기기:', self.device)
        else:
            self.device = torch.device('cpu')
            print('학습을 진행하는 기기:', self.device)

        self.FeatureExtractorBYOL = callAnyResnet(modelType=self.modelType,
                                                  numClass=self.embedSize,
                                                  normalizing=self.normalizing,
                                                  useLinLayer=self.useLinLayer
                                                  )

        try:
            print(f'loading {self.FESaveLoadDir} {self.FELoadNum}.pt')
            modelStateDict = torch.load(self.FESaveLoadDir +str(self.FELoadNum)+'.pt')
            missing = self.FeatureExtractorBYOL.load_state_dict(modelStateDict)
            print(f'missing : ',set(missing[1]))
            # assert (set(missing[1]) == {
            #     'contrastive_head.0.weight', 'contrastive_head.0.bias',
            #     'contrastive_head.2.weight', 'contrastive_head.2.bias'}
            #         or set(missing[1]) == {
            #             'contrastive_head.weight', 'contrastive_head.bias'})
            print(f'loading {self.FESaveLoadDir}{self.FELoadNum}.pt complete successfully!~!')
        except:
            if basemodelLoadName not in ['cifar10','cifar100','stl10']:
                print(f'loading base model..')
                loadPretrained4imagenet(baseLoadDir = self.basemodelSaveLoadDir+self.basemodelLoadName,
                                        model=self.FeatureExtractorBYOL)
                print(f'loading base model {self.basemodelSaveLoadDir + self.basemodelLoadName} complete!')
                self.FELoadNum = 0
            else:
                print(f'loading base model..')
                modelStateDict = torch.load(self.basemodelSaveLoadDir + self.basemodelLoadName)
                missing = self.FeatureExtractorBYOL.load_state_dict(modelStateDict,strict=False)
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
        except:
            print('loading saved head failed so start with fresh head')
            self.ClusterHead = myMultiCluster4SCAN(inputDim=self.embedSize,
                                                   dim1=self.cDim1,
                                                   nClusters=self.clusterNum,
                                                   numHead=self.numHeads,
                                                   isOutputProb=self.isInputProb,
                                                   layerMethod=self.layerMethod)
            self.headLoadNum = 0



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
                                      weight_decay=self.wDecay)

        self.optimizerCHead = Adam(self.ClusterHead.parameters(),
                                   lr=self.lr,
                                   weight_decay=self.wDecay)

        self.headOnlyConsisLossLst = []
        self.headOnlyConsisLossLstAvg = []
        self.headOnlyEntropyLossLst = []
        self.headOnlyEntropyLossLstAvg = []
        self.headOnlyTotalLossLst = []
        self.headOnlyTotalLossLstAvg = []
        self.clusterOnlyAccLst = []

        self.jointTrainingLossLst = []
        self.jointTrainingLossLstAvg = []
        self.jointTrainingAccLst = []

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

    def saveNearestNeighbor(self):

        self.FeatureExtractorBYOL.to(self.device)
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

        self.FeatureExtractorBYOL.to('cpu')
        self.ClusterHead.to('cpu')

    def trainHeadOnly(self,iterNum):

        self.FeatureExtractorBYOL.to(self.device)
        self.ClusterHead.to(self.device)

        self.FeatureExtractorBYOL.eval()

        indices = np.load(self.NNSaveDir+'NNs.npy')
        trainDataset = getCustomizedDataset4SCAN(downDir=self.downDir,
                                                 dataType=self.dataType,
                                                 transform=self.scanTransform,
                                                 nnNum= self.nnNum,
                                                 indices=indices,
                                                 toNeighborDataset=True)


        trainDataloader = DataLoader(trainDataset,shuffle=True,batch_size=self.trnBSize,num_workers=2)
        self.ClusterHead.train()
        for iter in range(iterNum):
            print(f'{iter}/{iterNum} training Start...')
            totalLossDict,\
            consistencyLossDict,\
            entropLossDict = scanTrain(train_loader= trainDataloader,
                                       featureExtractor = self.FeatureExtractorBYOL,
                                       headNum=self.numHeads,
                                       ClusterHead=self.ClusterHead,
                                       criterion=SCANLoss(entropyWeight=self.entropyWeight,
                                                          isInputProb=self.isInputProb),
                                       accumulNum= self.accumulNum,
                                       optimizer=[self.optimizerBackbone,self.optimizerCHead],
                                       device=self.device,
                                       update_cluster_head_only=self.update_cluster_head_only)

            for h in range(self.numHeads):
                self.clusterOnlyTotalLossDictPerHead[f'head_{h}'].append(np.mean(totalLossDict[f'head_{h}']))
                self.clusterOnlyClusteringLossDictPerHead[f'head_{h}'].append(np.mean(consistencyLossDict[f'head_{h}']))
                self.clusterOnlyEntropyLossDictPerHead[f'head_{h}'].append(np.mean(entropLossDict[f'head_{h}']))
            print(f'{iter}/{iterNum} training Complete !!!')

        self.ClusterHead.eval()

        self.FeatureExtractorBYOL.to('cpu')
        self.ClusterHead.to('cpu')

    def valHeadOnly(self):

        self.FeatureExtractorBYOL.to(self.device)
        self.ClusterHead.to(self.device)

        self.FeatureExtractorBYOL.eval()
        self.ClusterHead.eval()

        lst4CheckMinLoss = []
        for h in range(self.numHeads):
            lst4CheckMinLoss.append(np.mean(self.clusterOnlyTotalLossDictPerHead[f'head_{h}']))
        print(f'flushing cluster Loss lst complete')

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

        self.FeatureExtractorBYOL.to('cpu')
        self.ClusterHead.to('cpu')

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
        self.valHeadOnly()
        self.valHeadOnlyEnd()

    def trainJointly(self,iterNum=1):

        self.FeatureExtractorBYOL.to(self.device)
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
                                           featureExtractor = self.FeatureExtractorBYOL,
                                           headNum=self.numHeads,
                                           ClusterHead=self.ClusterHead,
                                           criterion=selfLabelLoss(selfLabelThreshold=self.selfLabelThreshold,
                                                                   isInputProb=self.isInputProb),
                                           optimizer=[self.optimizerBackbone,self.optimizerCHead],
                                           device=self.device,
                                           accumulNum=self.accumulNum,
                                           update_cluster_head_only=self.update_cluster_head_only)


            for h in range(self.numHeads):
                self.jointTrainingLossDictPerHead[f'head_{h}'].append(np.mean(totalLossDict[f'head_{h}']))

            print(f'{iter}/{iterNum} training Complete !!!')

        self.ClusterHead.eval()

        self.FeatureExtractorBYOL.to('cpu')
        self.ClusterHead.to('cpu')

    def jointVal(self):

        self.FeatureExtractorBYOL.to(self.device)
        self.ClusterHead.to(self.device)

        self.FeatureExtractorBYOL.eval()
        self.ClusterHead.eval()

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
                embededInput = self.FeatureExtractorBYOL(inputsRaw.to(self.device))

                clusterProb = self.ClusterHead.forwardWithMinLossHead(embededInput,
                                                                      headIdxWithMinLoss=self.minHeadIdxJointTraining)
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

            if modelLabelPerCluster[eachPredictedLabel] == eachGroundTruthLabel:
                accCheck.append(1)
            else:
                accCheck.append(0)

        print(f'len of accCheck is : {len(accCheck)}')
        self.jointTrainingAccLst.append(np.mean(accCheck))
        print(f'validation step end with accuracy : {np.mean(accCheck)}, total OK is : {np.sum(accCheck)} and '
              f'not OK is : {np.sum(np.array(accCheck) == 0)} with '
              f'length of data : {len(accCheck)}')

        self.FeatureExtractorBYOL.to('cpu')
        self.ClusterHead.to('cpu')

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
        # ax1.set_title('Head Only Train Loss clustering')

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(range(len(self.jointTrainingAccLst)), self.jointTrainingAccLst)
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('self labeling acc')
        # ax3.set_title(f'val acc , noise ratio : {self.labelNoiseRatio}')

        plt.savefig(self.plotSaveDir + 'selfLabelingResult.png', dpi=200)
        print('saving self labeling plot complete !')
        plt.close()
        plt.clf()
        plt.cla()

        with open(self.plotSaveDir + 'minLossHeadIdxJointTraining.pkl', 'wb') as F:
            pickle.dump(self.minHeadIdxLstJointTraining, F)
        print('saving head idx of having minimum loss lst complete')

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
        torch.save(self.FeatureExtractorBYOL.state_dict(), self.FESaveLoadDir + str(iteredNum + self.FELoadNum) + '.pt')
        print(f'saving head complete!!!')
        print(f'saving head complete!!!')
        print(f'saving head complete!!!')


    def checkConfidence(self):

        self.FeatureExtractorBYOL.to(self.device)
        self.ClusterHead.to(self.device)


        self.FeatureExtractorBYOL.eval()
        self.ClusterHead.eval()

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
                embededInput = self.FeatureExtractorBYOL(inputsRaw.to(self.device))
                clusterProb = self.ClusterHead.forwardWithMinLossHead(embededInput, headIdxWithMinLoss=self.minHeadIdx).cpu()

                clusterPred = torch.argmax(clusterProb, dim=1)
                clusterPredValue = torch.max(clusterProb,dim=1).values.cpu()

                clusterPredResult.append(clusterPred)
                clusterPredValueResult.append(clusterPredValue)
                gtLabelResult.append(loadedBatch['label'])

                # time.sleep(5)

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
            # print(eachCluster,modelLabelPerCluster[eachCluster])

        accCheck = dict()
        for eachCheckElement in noisedLabels4AccCheck:
            eachPredictedLabel = eachCheckElement[0].item()
            eachGroundTruthLabel = eachCheckElement[1]
            eachPredictValue = eachCheckElement[3]
            # if modelLabelPerCluster[eachPredictedLabel].size(0) != 0:

            if modelLabelPerCluster[eachPredictedLabel] == eachGroundTruthLabel:
                accResult = 1
            else:
                accResult = 0

            accCheck[eachPredictValue] = accResult

        finalConf, finalAcc,finalAllocNum = getAccPerConfLst(accCheck,10,minConf=0.95)

        plt.bar(finalConf,finalAcc)
        plt.xlabel('Conf Range')
        plt.ylabel('Acc')
        plt.savefig(self.plotSaveDir+'accPerConf.png',dpi=200)
        plt.close()
        plt.cla()
        plt.clf()

        plt.bar(finalConf, finalAllocNum)
        plt.xlabel('Conf Range')
        plt.ylabel('Allocated Num')
        plt.savefig(self.plotSaveDir + 'AllocPerConf.png', dpi=200)
        plt.close()
        plt.cla()
        plt.clf()

        self.FeatureExtractorBYOL.to('cpu')
        self.ClusterHead.to('cpu')

    def saveFiltered(self):

        self.FeatureExtractorBYOL.to(self.device)
        self.ClusterHead.to(self.device)

        self.FeatureExtractorBYOL.eval()
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

                embededInput = self.FeatureExtractorBYOL(inputsRaw.to(self.device))
                clusterProb = self.ClusterHead.forwardWithMinLossHead(embededInput,
                                                                      headIdxWithMinLoss=self.minHeadIdx).cpu()
                clusterPred = torch.argmax(clusterProb, dim=1)
                clusterPredValue = torch.max(clusterProb, dim=1).values.cpu()

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

        with open(self.plotSaveDir+f'filteredData.pkl','wb') as F:
            pickle.dump(finalDict,F)

        print('saving confident data indices and cluster complete ')

        self.FeatureExtractorBYOL.to('cpu')
        self.ClusterHead.to('cpu')


    def saveNoiseDataIndices(self,theNoise):

        self.FeatureExtractorBYOL.to(self.device)
        self.ClusterHead.to(self.device)

        self.FeatureExtractorBYOL.eval()
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

                embededInput = self.FeatureExtractorBYOL(inputsRaw.to(self.device))
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
                noisedLabels4AccCheck.append([eachClusterPred.cpu(),
                                              eachGtLabel.cpu(),
                                              noisedLabel,
                                              eachClusterPredValue,
                                              eachIndex])
                noiseOrNot[eachIndex] = True
            else:
                noisedLabels.append(eachGtLabel)
                noiseOrNot[eachIndex] = False

        noisedLabels = torch.cat(noisedLabels)
        noisedDataDict = {
            'resultLst' : noisedLabels4AccCheck,
            'noiseOrNot' : noiseOrNot
        }

        with open(self.plotSaveDir+f'noisedDataOnly_{str(theNoise)}.pkl', 'wb') as F:
            pickle.dump(noisedDataDict,F)
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



    def loadModel4filtered(self,nclass):

        self.FeatureExtractorBYOL.to('cpu')
        self.ClusterHead.to('cpu')

        self.nClass =nclass

        self.FeatureExtractor4FTed = callAnyResnet(modelType=self.modelType,
                                                      numClass=self.embedSize,
                                                      normalizing=False,
                                                      useLinLayer=False,
                                                      )


        print(f'loading {self.FTedFESaveLoadDir} {self.FTedFELoadNum}.pt')
        modelStateDict = torch.load(self.FTedFESaveLoadDir + str(self.FTedFELoadNum) + '.pt')
        missing = self.FeatureExtractor4FTed.load_state_dict(modelStateDict)
        print(f'missing : ', set(missing[1]))
        # assert (set(missing[1]) == {
        #     'contrastive_head.0.weight', 'contrastive_head.0.bias',
        #     'contrastive_head.2.weight', 'contrastive_head.2.bias'}
        #         or set(missing[1]) == {
        #             'contrastive_head.weight', 'contrastive_head.bias'})
        print(f'loading {self.FTedFESaveLoadDir}{self.FTedFELoadNum}.pt complete successfully!~!')


        self.ClusterHeadFTed = myPredictorHead(inputDim=self.embedSize,
                                           dim1=self.cDim1,
                                           nClass=self.nClass)

        headLoadedDict = torch.load(self.FTedheadSaveLoadDir + str(self.FTedheadLoadNum) + '.pt')
        self.ClusterHeadFTed.load_state_dict(headLoadedDict)
        print(f'loading saved head : {str(self.FTedheadLoadNum)}.pt complete!!!!!!!')

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

    def trainFilteredDataNaiveVer(self):

        self.FeatureExtractor4FTed.to(self.device)
        self.ClusterHeadFTed.to(self.device)

        trainDataset = filteredDatasetNaive4SCAN(downDir=self.downDir,
                                                 savedIndicesDir = self.plotSaveDir,
                                                 dataType=self.dataType,
                                                 noiseRatio = self.labelNoiseRatio,
                                                 transform= self.baseTransform
                                                 )

        trainDataloader = DataLoader(trainDataset, shuffle=True, batch_size=self.jointTrnBSize, num_workers=2)

        self.ClusterHeadFTed.train()
        for iter in range(iterNum):
            print(f'{iter}/{iterNum} training Start...')
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

            print(f'{iter}/{iterNum} training Complete !!!')

        self.ClusterHeadFTed.eval()

        self.FeatureExtractor4FTed.to('cpu')
        self.ClusterHeadFTed.to('cpu')

    def valFilteredDataNaiveVer(self):

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

        # result of prediction for each inputs
        labelPredResult = torch.cat(labelPredResult)
        # for eachLabelUnique in torch.unique(labelPredResult):
        #     print(
        #         f' {torch.count_nonzero(labelPredResult == eachLabelUnique)} '
        #         f'allocated for cluster :{eachLabelUnique}'
        #         f'when validating')

        # ground truth label for each inputs
        gtLabelResult = torch.cat(gtLabelResult).unsqueeze(1)

        assert gtLabelResult.size() == labelPredResult.size()
        print(f'size of gtLabelResult is : {gtLabelResult.size()} and preResult is : {labelPredResult.size()}')
        # print(f'clusterPred has size : {clusterPredResult.size()} , gtLabelResult has size : {gtLabelResult.size()}')

        acc4Total = torch.mean((gtLabelResult==labelPredResult).float())

        acc4noiseOnly = []
        trainDataset = noisedOnlyDatasetNaive4SCAN(downDir=self.downDir,
                                                   savedIndicesDir = self.plotSaveDir,
                                                   dataType=self.dataType,
                                                   noiseRatio = self.labelNoiseRatio,
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

        # result of prediction for each inputs
        labelPredResult = torch.cat(labelPredResult)
        # ground truth label for each inputs
        gtLabelResult = torch.cat(gtLabelResult).unsqueeze(1)

        assert labelPredResult.size() == gtLabelResult.size()

        acc4noiseOnly = torch.mean((gtLabelResult==labelPredResult).float())

        print(f'acc total is :{acc4Total} and acc noise only is : {acc4noiseOnly}')

        self.ftedValAcc4TotalLst.append(acc4Total)
        self.ftedValAcc4NoiseOnlyLst.append(acc4noiseOnly)

        self.FeatureExtractor4FTed.to('cpu')
        self.ClusterHeadFTed.to('cpu')

    def valFilteredDataNaiveVerEnd(self):

        fig = plt.figure(constrained_layout=True)

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(range(len(self.ftedTrainLossLst)), self.ftedTrainLossLst)
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('filtered loss')
        # ax1.set_title('Head Only Train Loss clustering')

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(range(len(self.ftedTrainAccLst)), self.ftedTrainAccLst)
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('filtered train acc')
        # ax3.set_title(f'val acc , noise ratio : {self.labelNoiseRatio}')

        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(range(len(self.ftedValAcc4TotalLst)), self.ftedValAcc4TotalLst)
        ax3.set_xlabel('epoch')
        ax3.set_ylabel('filtered total loss')
        # ax1.set_title('Head Only Train Loss clustering')

        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(range(len(self.ftedValAcc4NoiseOnlyLst)), self.ftedValAcc4NoiseOnlyLst)
        ax4.set_xlabel('epoch')
        ax4.set_ylabel('filtered noiseonly acc')
        # ax3.set_title(f'val acc , noise ratio : {self.labelNoiseRatio}')

        plt.savefig(self.plotSaveDir + 'ftedTrainingResult.png', dpi=200)
        print('saving filtered training plot complete !')
        plt.close()
        plt.clf()
        plt.cla()

    def executeFTedTraining(self):

        self.trainFilteredDataNaiveVer()
        self.valFilteredDataNaiveVer()
        self.valFilteredDataNaiveVerEnd()










































                
                




        




