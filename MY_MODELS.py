"""
    In the case of code for resnet model(BasicBlock,Bottleneck,ResNet),

    @inproceedings{vangansbeke2020scan,
    title={Scan: Learning to classify images without labels},
    author={Van Gansbeke, Wouter and Vandenhende, Simon and Georgoulis, Stamatios and Proesmans, Marc and Van Gool, Luc},
    booktitle={Proceedings of the European Conference on Computer Vision},
    year={2020}
}
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter
import math
from torchvision import models





class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        return out

# load resnet for non imagnet dataset
# cifar10, cifar20, stl10
class callAnyResnet(nn.Module):
    def __init__(self,modelType,numClass,headType='mlp',useLinLayer=False,normalizing=True):
        super(callAnyResnet, self).__init__()

        self.useLinLayer = useLinLayer
        self.normalizing = normalizing

        if modelType == 'resnet18':
            self.backbone= ResNet(block=BasicBlock,
                                  num_blocks=[2,2,2,2]
                                  )
            self.backboneDim= 512
        elif modelType == 'resnet34':
            self.backbone = ResNet(block=BasicBlock,
                                   num_blocks=[3,4,6,3]
                                   )
            self.backboneDim = 512
        elif modelType == 'resnet50':
            self.backbone = models.resnet50(pretrained=False)
            self.backbone.fc = nn.Identity()
            self.backboneDim = 2048

        else:
            self.backbone = ResNet(block=BasicBlock,
                                   num_blocks=[2, 2, 2, 2]
                                   )
            self.backboneDim = 512

        if useLinLayer ==True:
            if headType == 'oneLinear':
                self.contrastive_head = nn.Linear(self.backboneDim,numClass)
            elif headType == 'mlp':
                self.contrastive_head = nn.Sequential(nn.Linear(self.backboneDim, self.backboneDim),
                                          nn.ReLU(),
                                          nn.Linear(self.backboneDim, numClass))


    def forward(self,x):
        if self.normalizing:
            out = F.normalize(self.backbone(x),dim=1)
        else:
            out = self.backbone(x)
        if self.useLinLayer == True:
            out =  self.contrastive_head(out)
            out = F.normalize(out,dim=1)
        return out

# code for loading resnet for imagenet data
# imagenet10
class callResnet4Imagenet(nn.Module):
    def __init__(self,modelType,numClass,headType='mlp',useLinLayer=False,normalizing=True):
        super(callResnet4Imagenet, self).__init__()

        self.useLinLayer = useLinLayer
        self.normalizing = normalizing

        if modelType == 'resnet18':
            self.backbone= ResNet(block=BasicBlock,
                                  num_blocks=[2,2,2,2]
                                  )
            self.backboneDim= 512
        elif modelType == 'resnet34':
            self.backbone = ResNet(block=BasicBlock,
                                   num_blocks=[3,4,6,3]
                                   )
            self.backboneDim = 512
        elif modelType == 'resnet50':
            self.backbone = ResNet(block=Bottleneck,
                                   num_blocks=[3,4,6,3]
                                   )
            self.backboneDim = 1024

        else:
            self.backbone = ResNet(block=BasicBlock,
                                   num_blocks=[2, 2, 2, 2]
                                   )
            self.backboneDim = 512

        if useLinLayer ==True:
            if headType == 'oneLinear':
                self.contrastive_head = nn.Linear(self.backboneDim,numClass)
            elif headType == 'mlp':
                self.contrastive_head = nn.Sequential(nn.Linear(self.backboneDim, self.backboneDim),
                                          nn.ReLU(),
                                          nn.Linear(self.backboneDim, numClass))



# code for cluster head
class myCluster4SCAN(nn.Module):
    def __init__(self,
                 inputDim,
                 dim1,
                 nClusters,
                 lossMethod='CE',
                 isOutputProb= False,
                 layerMethod='linear'):
        super(myCluster4SCAN, self).__init__()

        self.inputDim = inputDim
        self.dim1 = dim1
        self.nClusters = nClusters
        self.layerMethod= layerMethod
        self.isOutputProb= isOutputProb

        if self.layerMethod == 'linear':
            self.MLP = nn.Sequential(
                nn.Linear(in_features=self.inputDim,out_features=self.nClusters)
            )
        if self.layerMethod == 'mlp':
            self.MLP = nn.Sequential(
                nn.Linear(in_features=self.inputDim, out_features=self.dim1),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=self.dim1, out_features=self.nClusters)
            )

        self.lossMethod = lossMethod
        if self.lossMethod == 'CE':
            self.LOSS = nn.CrossEntropyLoss()


    def getLoss(self,pred,label,withEntropy=False,entropyWeight=5.0,clusteringWeight=1.0):

        if withEntropy ==True:
            totalLoss = [clusteringWeight * self.LOSS(pred,label), entropyWeight * self.calEntropy(actions=pred)]

        else:
            totalLoss = self.LOSS(pred,label)

        return totalLoss

    def forward(self,x):

        if self.isOutputProb ==False:
            out = self.MLP(x)
        else:
            out = F.softmax(self.MLP(x), dim=1)

        return out



# code for cluster head when training with
# high confident data only
class myPredictorHead(nn.Module):
    def __init__(self,
                 inputDim,
                 dim1,
                 nClass
                 ):
        super(myPredictorHead, self).__init__()

        self.inputDim = inputDim
        self.dim1 = dim1
        self.nClass = nClass

        self.MLP = nn.Sequential(
            nn.Linear(in_features=self.inputDim, out_features=self.dim1),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.dim1, out_features=self.nClass)
        )

    def forward(self,x):

        out = self.MLP(x)
        return out

# multi cluster head code when training with SCAN
class myMultiCluster4SCAN(nn.Module):
    def __init__(self,
                 inputDim,
                 dim1,
                 nClusters,
                 numHead,
                 lossMethod='CE',
                 isOutputProb=False,
                 layerMethod='linear'):
        super(myMultiCluster4SCAN, self).__init__()

        self.inputDim = inputDim
        self.dim1 = dim1
        self.nClusters = nClusters
        self.numHead = numHead
        self.isOutputProb= isOutputProb
        self.lossMethod = lossMethod
        self.layerMethod = layerMethod

        for h in range(self.numHead):
            headH = myCluster4SCAN(inputDim=self.inputDim,
                                    dim1=self.dim1,
                                    nClusters=self.nClusters,
                                    lossMethod=self.lossMethod,
                                    isOutputProb = self.isOutputProb,
                                    layerMethod=self.layerMethod)
            self.__setattr__(f'eachHead_{h}',headH)


    def forward(self,x,inputDiff=True,headIdxWithMinLoss=None):

        if headIdxWithMinLoss ==None:

            totalForwardResult = {}

            if inputDiff == True:
                for h in range(self.numHead):
                    eachForwardResult = self.__getattr__(f'eachHead_{h}').forward(x[h])
                    totalForwardResult[f'eachHead_{h}'] = eachForwardResult
            else:
                for h in range(self.numHead):
                    eachForwardResult = self.__getattr__(f'eachHead_{h}').forward(x)
                    totalForwardResult[f'eachHead_{h}'] = eachForwardResult

            return totalForwardResult

        else:
            return self.__getattr__(f'eachHead_{headIdxWithMinLoss}').forward(x=x)



    def getTotalLoss(self,x,label,withEntropy=False,entropyWeight=5.0,clusteringWeight=1.0):

        totalLoss = {}

        for h in range(self.numHead):
            eachLossH = self.__getattr__(f'eachHead_{h}').getLoss(pred=x[h],
                                                                  label=label[h],
                                                                  withEntropy=withEntropy,
                                                                  entropyWeight=entropyWeight,
                                                                  clusteringWeight=clusteringWeight)
            totalLoss[f'eachHead_{h}'] = eachLossH
        print(f'total loss is : {totalLoss}')
        return totalLoss

    def forwardWithMinLossHead(self,inputs,headIdxWithMinLoss):

        return self.__getattr__(f'eachHead_{headIdxWithMinLoss}').forward(x=inputs)


