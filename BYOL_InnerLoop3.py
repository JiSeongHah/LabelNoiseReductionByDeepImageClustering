import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.optim import Adam,AdamW,SGD
from byol_pytorch import BYOL
from torchvision import models
from MY_MODELS import callAnyResnet
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import models
from tqdm import tqdm
import time
from save_funcs import mk_name,createDirectory
import os


class doBYOL(nn.Module):
    def __init__(self,
                 modelType,
                 L2NormalEnd,
                 trnBSize,
                 gpuUse,
                 saveRange,
                 embedSize,
                 doSimSiam,
                 plotSaveDir,
                 whichOptim,
                 modelSaveDir,
                 wDecay=0.001):
        super(doBYOL,self).__init__()


        self.modelType = modelType
        self.L2NormalEnd = L2NormalEnd
        self.trnBSize = trnBSize
        self.gpuUse = gpuUse
        self.saveRange = saveRange
        self.embedSize = embedSize
        self.doSimSiam = doSimSiam
        self.whichOptim = whichOptim
        self.wDecay = wDecay
        self.plotSaveDir = plotSaveDir
        self.modelSaveDir = modelSaveDir


        self.modelBYOL = callAnyResnet(modelType=self.modelType,
                                       numClass = self.embedSize,
                                       L2NormalEnd=self.L2NormalEnd)

        if self.doSimSiam == True:

            self.learner = BYOL(self.modelBYOL,
                                image_size=32,
                                hidden_layer='avgpool',
                                use_momentum = self.doSimSiam
                                )
        else:
            self.learner = BYOL(self.modelBYOL,
                                image_size=32,
                                hidden_layer='avgpool'
                                )

        if self.whichOptim == 'adam':
            self.optimizer = Adam(self.learner.parameters(), weight_decay=self.wDecay, lr=3e-4)
        elif self.whichOptim == 'adamw':
            self.optimizer = AdamW(self.learner.parameters(), weight_decay=self.wDecay,lr=3e-4)
        else:
            self.optimizer = SGD(self.learner.parameters(), lr=3e-4)

        transform = transforms.Compose([transforms.ToTensor()])
        self.dataset = CIFAR10(root='~/', train=True, download=True, transform=transform)
        self.trainDataloader = DataLoader(self.dataset,
                                          batch_size=self.trnBSize,
                                          shuffle=True,
                                          num_workers=2)

        if self.gpuUse == True:
            USE_CUDA = torch.cuda.is_available()
            print(USE_CUDA)
            self.device = torch.device('cuda' if USE_CUDA else 'cpu')
            print('학습을 진행하는 기기:', self.device)
        else:
            self.device = torch.device('cpu')
            print('학습을 진행하는 기기:', self.device)


        self.lossLst = []
        self.lossLstAvg = []

        self.learner.to(self.device)

    def trnLearner(self,iterNum):

        TDataLoader = tqdm(self.trainDataloader)

        globalTime = time.time()


        for idx, (inputs, label) in enumerate(TDataLoader):
            localTime = time.time()

            inputs = inputs.to(self.device)
            loss = self.learner(inputs)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lossLst.append(loss.cpu().item())
            if self.doSimSiam == False:
                self.learner.update_moving_average()

            localTimeElaps = round(time.time() - localTime,2)
            globalTimeElaps = round(time.time() - globalTime,2)

            TDataLoader.set_description(f'Processing : {idx} of {iterNum}')
            TDataLoader.set_postfix(Gelapsed=globalTimeElaps,
                                    Lelapsed=localTimeElaps,
                                    loss=loss.item())


            # plt.plot(range(len(self.lossLst)), self.lossLst)
            # plt.show()
            # plt.close()



    def trnMultipleTime(self,iterNum):

        for eachEpoch in range(iterNum):
            self.trnLearner(iterNum=eachEpoch)

            self.lossLstAvg.append(np.mean(self.lossLst))
            self.flushLst()

            plt.plot(range(len(self.lossLstAvg)), self.lossLstAvg)
            plt.savefig(self.plotSaveDir+'lossPlot.png')
            plt.close()

            if eachEpoch % self.saveRange == 0:
                torch.save(self.modelBYOL.state_dict(),self.modelSaveDir+str(eachEpoch)+'.pt')

    def flushLst(self):

        self.lossLst.clear()
        print('flushing lst complete')


os.environ['CUDA_VISIBLE_DEVICES'] = "3"

L2NormalEnd= True
modelType = 'resnet18'
trnBSize = 4096
gpuUse = True
iterNum = 100000
embedSize = 512
doSimSiam = False
saveRange = 50
whichOptim = 'adamw'
wDecay = 0.0005
specificName = mk_name(dirBYOL='/',
                       modelType=modelType,
                       L2NormalEnd=L2NormalEnd,
                       optim=whichOptim,
                       wDecay=wDecay,
                       trnBSize=trnBSize,
                       embedSize=embedSize,
                       doSimSiam=doSimSiam)

baseDir = '/home/a286/hjs_dir1/BYOL_RESULT/'

plotSaveDir = baseDir + specificName + '/'
modelSaveDir = baseDir + specificName + '/MODELS/'
createDirectory(modelSaveDir)


myBYOL = doBYOL(modelType=modelType,
                L2NormalEnd=L2NormalEnd,
                trnBSize=trnBSize,
                gpuUse=gpuUse,
                saveRange=saveRange,
                embedSize=embedSize,
                doSimSiam=doSimSiam,
                whichOptim=whichOptim,
                plotSaveDir=plotSaveDir,
                modelSaveDir=modelSaveDir
                )
myBYOL.trnMultipleTime(iterNum=iterNum)

