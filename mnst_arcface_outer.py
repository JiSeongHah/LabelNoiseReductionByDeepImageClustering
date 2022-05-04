import copy

import torch
from torch.optim import AdamW, Adam
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import models
from CALAVG import cal_avg_error
from MY_MODELS import BasicBlock, ResNet, ArcMarginProduct
from save_funcs import mk_name,lst2csv,createDirectory
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from mnst_arcface_innerPredictor import innerPredictor
from mnst_arcface_innerTestPredictor import innerTestPredictor
from LoadInnerPredictor import func_LoadInnerPredictor
from mnst_arcface_dataloading import func_getData
from MK_NOISED_DATA import get_noisy_data, mk_noisy_label


class OuterLoop(nn.Module):
    def __init__(self,
                 modelSaveLoadDir,
                 plotSaveDir,
                 innermodelLoadNum,
                 dataDir,
                 bSizeTrn,
                 bSizeVal,
                 LinNum,
                 s,
                 m,
                 gpuUse,
                 MaxStepTrn,
                 MaxStepVal,
                 iterToAccumul,
                 beta4f1=100
                 ):
        super(OuterLoop, self).__init__()

        self.modelSaveLoadDir = modelSaveLoadDir
        self.plotSaveDir = plotSaveDir
        self.innermodelLoadNum = innermodelLoadNum


        self.MyInnerPredictor = func_LoadInnerPredictor(modelSaveLoadDir = modelSaveLoadDir,
                                                        plotSaveDir=plotSaveDir,
                                                        innermodelLoadNum= innermodelLoadNum,
                                                        bSizeTrn = bSizeTrn,
                                                        bSizeVal = bSizeVal,
                                                        LinNum = LinNum,
                                                        s = s,
                                                        m = m,
                                                        gpuUse = gpuUse,
                                                        
                                                        MaxStepTrn = MaxStepTrn,
                                                        MaxStepVal = MaxStepVal,
                                                        iterToAccumul = iterToAccumul,
                                                        beta4f1=beta4f1
                                                        )

        self.dataDir = dataDir

    def trainInnerPredictor(self,
                            totalTInput,
                            totalTLabel,
                            totalVInput,
                            totalVLabel,
                            iterNum
                            ):

        self.MyInnerPredictor.setDataLoader(trnInput = totalTInput,
                                            trnLabel = totalTLabel,
                                            valInput = totalVInput,
                                            valLabel = totalVLabel)
        
        for i in range(iterNum):
            self.MyInnerPredictor.FIT(iterationNum=iterNum,stopThreshold = 1e-2)
            torch.save(self.MyInnerPredictor,self.modelSaveLoadDir+str((i+1)*iterNum+self.innermodelLoadNum)+'.pth')
            print(f'saving {(i+1)*iterNum}th inner model complete')




    def getDataValue4FeatureNoise(self,data4Criterion,data2Compare):

        featureCriterion,labelCriterion = self.MyInnerPredictor.getFeatures(data4Criterion)
        featureCompare,labelCompare = self.MyInnerPredictor.getFeatures(data2Compare)

        cosineTensor = F.linear(featureCompare, featureCriterion)
        meanCosineTensor = torch.mean(cosineTensor,dim=1)

        Datavalue = meanCosineTensor

        return Datavalue

    def getFilteredLowDataValueFeatureNoise(self,threshold,data4Criterion,data2Compare):

        DataValue = self.getDataValue4FeatureNoise(data4Criterion=data4Criterion,
                                                   data2Compare=data2Comapre
                                                   )

        DatavalueAssumedHelpful = DataValue > threshold

        filteredTensor = data2Compare[0][DatavalueAssumedHelpful]
        filteredLabel = data2Compare[1][DatavalueAssumedHelpful]

        return filteredTensor,filteredLabel

    def getDataValue4LabelNoise(self, data4Criterion, data2Compare,label2Check,k):

        featureCriterion,labelCriterion = self.MyInnerPredictor.getFeatures(data4Criterion[0],data4Criterion[1])
        featureCompare,labelCompare = self.MyInnerPredictor.getFeatures(data2Compare[0],data2Compare[1])

        cosineTensor = F.linear(featureCompare,featureCriterion)
        topKIndices = torch.topk(cosineTensor,dim=1,k=k).indices

        inputTensor4gather = labelCriterion.repeat(topKIndices.size(0),1)

        totalLabelTensor = torch.gather(input=inputTensor4gather,dim=1,index=topKIndices) == label2Check
        DataValue = torch.sum(totalLabelTensor.long(),dim=1) / k

        return DataValue

    def getFilteredLowDataValueLabelNoise(self,threshold,data2Compare,data2Criterion,label2Check,k):

        datavalue = self.getDataValue4LabelNoise(data4Criterion=data2Criterion,
                                                 data2Compare=data2Compare,
                                                 label2Check=label2Check,
                                                 k=k)

        # if element is true : means that element is assumed as label2check
        filteredIndex = torch.ge(datavalue,threshold)

        # filteredInput = data2Compare[0][filteredIndex]
        # filteredLabel = data2Compare[1][filteredIndex]

        return filteredIndex

    def comapareLabelAndPred(self,oriLabel,predLabel):

        compared = torch.eq(oriLabel,predLabel).long()

        accuracy = torch.sum(compared)/compared.size()

        return accuracy

    def executeFeatureNoiseVer1(self,
                               usePretrained,
                               threshold,
                               splitRatio,
                               iterNum=None):

        TdataRest, \
        TlabelRest, \
        TdataZero, \
        TlabelZero, \
        VdataRest, \
        VlabelRest, \
        VdataZero, \
        VlabelZero = func_getData(self.dataDir)

        datawithoutNoise, datawithNoise, labelRaw,labelNoisePart = get_noisy_data(wayofdata='sum',
                                                                                  splitRatio=splitRatio,
                                                                                  RL_train_data_zero=TdataZero,
                                                                                  RL_train_label_zero=TlabelZero)

        totalTInput = torch.cat((TdataRest, datawithoutNoise), dim=0)
        totalTLabel = torch.cat((TlabelRest, labelRaw), dim=0).long()
        totalVInput = torch.cat((VdataRest, VdataZero), dim=0)
        totalVLabel = torch.cat((VlabelRest, VlabelZero), dim=0).long()

        if usePretrained == False:
            self.trainInnerPredictor(totalTInput,
                                     totalTLabel,
                                     totalVInput,
                                     totalVLabel,
                                     iterNum)

        ##################### set predictors ################################
        Predictor4baseline = innerTestPredictor(plotSaveDir=self.plotSaveDir,
                                                ModelName='baseline_')
        Predictor2compare = copy.deepcopy(Predictor4baseline)
        ##################### set predictors ################################


        ######################### cal basline result ########################
        totalTInput = torch.cat((TdataRest, datawithoutNoise,datawithNoise), dim=0)
        totalTLabel = torch.cat((TlabelRest, labelRaw,labelNoisePart), dim=0).long()
        totalVInput = torch.cat((VdataRest, VdataZero), dim=0)
        totalVLabel = torch.cat((VlabelRest, VlabelZero), dim=0).long()

        Predictor4baseline.setDataLoader(trnInput = totalTInput,
                                            trnLabel = totalTLabel,
                                            valInput = totalVInput,
                                            valLabel = totalVLabel)
        Predictor4baseline.FIT(iterNum = iterNum,
                               stopThreshold= 1e-2)

        ACC_baseline = Predictor4baseline.GETSCORE()
        ######################### cal basline result ########################


        ######################### cal real result ########################
        filteredData,filteredLabel = self.getFilteredLowDataValueFeatureNoise(threshold=threshold,
                                                                              data4Criterion=[datawithoutNoise,labelRaw],
                                                                              data2Compare=[datawithNoise,labelNoisePart])
        totalTInput = torch.cat((TdataRest, datawithoutNoise,filteredData), dim=0)
        totalTLabel = torch.cat((TlabelRest, labelRaw,filteredLabel), dim=0).long()
        totalVInput = torch.cat((VdataRest, VdataZero), dim=0)
        totalVLabel = torch.cat((VlabelRest, VlabelZero), dim=0).long()

        Predictor2compare.setDataLoader(trnInput = totalTInput,
                                            trnLabel = totalTLabel,
                                            valInput = totalVInput,
                                            valLabel = totalVLabel)
        Predictor2compare.FIT(iterNum = iterNum,
                               stopThreshold= 1e-2)

        ACC_compare = Predictor2compare.GETSCORE()
        ######################### cal real result ########################


        return ACC_compare, ACC_baseline


    def executeLabelNoiseVer(self,
                             usePretrained,
                             threshold,
                             splitRatio,
                             k,
                             iterNum=None):
        
        TdataRest,\
        TlabelRest,\
        TdataZero,\
        TlabelZero,\
        VdataRest,\
        VlabelRest,\
        VdataZero,\
        VlabelZero = func_getData(self.dataDir)

        noisedDataZero, noisedLabelZero =  mk_noisy_label(raw_data= TdataZero,
                                                          raw_label = TlabelZero,
                                                          splitRatio=splitRatio)


        totalTInput = torch.cat((TdataRest,noisedDataZero),dim=0)
        totalTLabel = torch.cat((TlabelRest,noisedLabelZero),dim=0).long()

        totalVInput = torch.cat((VdataRest,VdataZero),dim=0)
        totalVLabel = torch.cat((VlabelRest,VlabelZero),dim=0).long()

        if usePretrained == False:
            self.trainInnerPredictor(totalTInput,
                                     totalTLabel,
                                     totalVInput,
                                     totalVLabel,
                                     iterNum)

        filteredIndex = self.getFilteredLowDataValueLabelNoise(threshold=threshold,
                                                               data2Compare=[noisedDataZero,noisedLabelZero],
                                                               data2Criterion=[totalTInput,totalTLabel],
                                                               label2Check=0,
                                                               k=k
                                                               )
        Acc = self.comapareLabelAndPred(oriLabel=TlabelZero,predLabel=filteredIndex) - splitRatio

        return Acc , filteredIndex, noisedLabelZero






innermodelLoadNum = 300
dataDir = '/home/a286winteriscoming/DVMETRIC/'
bSizeTrn = 128
bSizeVal = 128
LinNum = 512
s = 64
m = 0.5
gpuUse = True
MaxStepTrn = 1000000000000
MaxStepVal = 10000000000000
iterToAccumul = 2
beta4f1 = 100
iterNum = 100
specificName = mk_name(dir1='/',
                       s=s,
                       m=m,
                       beta4f1=beta4f1,
                       bSizeTrn=bSizeTrn)

modelSaveLoadDir = dataDir + specificName+ '/Models/'
plotSaveDir = dataDir + specificName+ '/PLOTS/'

createDirectory(modelSaveLoadDir)
createDirectory(plotSaveDir)

doit = OuterLoop(
    modelSaveLoadDir = modelSaveLoadDir,
    plotSaveDir=plotSaveDir,
    innermodelLoadNum = innermodelLoadNum,
    dataDir = dataDir,
    bSizeTrn = bSizeTrn,
    bSizeVal = bSizeVal,
    LinNum = LinNum,
    s =s,
    m =m,
    gpuUse = gpuUse,
    MaxStepTrn = MaxStepTrn,
    MaxStepVal = MaxStepVal,
    iterToAccumul = iterToAccumul,
    beta4f1 = beta4f1
)

doit.executeLabelNoiseVer(usePretrained=False,
                          threshold=0.9,
                          splitRatio= 0.8,
                          k=7,
                          iterNum=100)



































