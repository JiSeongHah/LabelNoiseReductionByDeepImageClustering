import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter
import math
from torchvision import models

class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # 첫번째층
        # ImgIn shape=(?, 28, 28, 1)
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 두번째층
        # ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 전결합층 7x7x64 inputs -> 10 outputs
        self.fc = torch.nn.Linear(7 * 7 * 64, 10, bias=True)

        # 전결합층 한정으로 가중치 초기화
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)  # 전결합층을 위해서 Flatten
        out = self.fc(out)
        return out




class BasicBlock(nn.Module):
    # mul은 추후 ResNet18, 34, 50, 101, 152등 구조 생성에 사용됨
    mul = 1

    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlock, self).__init__()


        # stride를 통해 너비와 높이 조정
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)

        # stride = 1, padding = 1이므로, 너비와 높이는 항시 유지됨
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        # x를 그대로 더해주기 위함
        self.shortcut = nn.Sequential()

        # 만약 size가 안맞아 합연산이 불가하다면, 연산 가능하도록 모양을 맞춰줌
        if stride != 1:  # x와
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)  # 필요에 따라 layer를 Skip
        out = F.relu(out)


        return out


class BottleNeck(nn.Module):
    # 논문의 구조를 참고하여 mul 값은 4로 지정, 즉, 64 -> 256
    mul = 4

    def __init__(self, in_planes, out_planes, stride=1):
        super(BottleNeck, self).__init__()


        # 첫 Convolution은 너비와 높이 downsampling
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)

        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.conv3 = nn.Conv2d(out_planes, out_planes * self.mul, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes * self.mul)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != out_planes * self.mul:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes * self.mul, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes * self.mul)
            )

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks,num_classes=10,mnst_ver=True):
        super(ResNet, self).__init__()
        # RGB 3개채널에서 64개의 Kernel 사용 (논문 참고)
        self.in_planes = 64

        # Resnet 논문 구조의 conv1 파트 그대로 구현
        if mnst_ver ==True:
            self.conv1 = nn.Conv2d(1, self.in_planes, kernel_size=7, stride=2, padding=3)
        else:
            self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Basic Resiudal Block일 경우 그대로, BottleNeck일 경우 4를 곱한다.
        self.linear = nn.Linear(512 * block.mul, num_classes)

    # 다양한 Architecture 생성을 위해 make_layer로 Sequential 생성
    def make_layer(self, block, out_planes, num_blocks, stride):
        # layer 앞부분에서만 크기를 절반으로 줄이므로, 아래와 같은 구조
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            layers.append(block(self.in_planes, out_planes, strides[i]))
            self.in_planes = block.mul * out_planes
        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.maxpool1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #print(out.size())
        out = self.avgpool(out)
        #print(out.size())
        out = torch.flatten(out, 1)
        out = self.linear(out)
        #print(out.size())


        return out


class BasicBlock4one(nn.Module):
    # mul은 추후 ResNet18, 34, 50, 101, 152등 구조 생성에 사용됨
    mul = 1

    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlock4one, self).__init__()


        # stride를 통해 너비와 높이 조정
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


        # stride = 1, padding = 1이므로, 너비와 높이는 항시 유지됨
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

        # x를 그대로 더해주기 위함
        self.shortcut = nn.Sequential()

        # 만약 size가 안맞아 합연산이 불가하다면, 연산 가능하도록 모양을 맞춰줌
        if stride != 1:  # x와

            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):

        out = self.conv1(x)


        out = F.relu(out)

        out = self.conv2(out)


        out += self.shortcut(x)  # 필요에 따라 layer를 Skip

        out = F.relu(out)


        return out


class BottleNeck4one(nn.Module):
    # 논문의 구조를 참고하여 mul 값은 4로 지정, 즉, 64 -> 256
    mul = 4

    def __init__(self, in_planes, out_planes, stride=1):
        super(BottleNeck4one, self).__init__()

        # 첫 Convolution은 너비와 높이 downsampling
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)


        self.conv3 = nn.Conv2d(out_planes, out_planes * self.mul, kernel_size=1, stride=1, bias=False)


        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != out_planes * self.mul:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes * self.mul, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes * self.mul)
            )

    def forward(self, x):

        out = self.conv1(x)

        out = F.relu(out)
        out = self.conv2(out)

        out = F.relu(out)
        out = self.conv3(out)

        out += self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet4one(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, mnst_ver=True):
        super(ResNet4one, self).__init__()
        # RGB 3개채널에서 64개의 Kernel 사용 (논문 참고)
        self.in_planes = 64

        # Resnet 논문 구조의 conv1 파트 그대로 구현
        if mnst_ver == True:
            self.conv1 = nn.Conv2d(1, self.in_planes, kernel_size=7, stride=2, padding=3)
        else:
            self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3)

        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Basic Resiudal Block일 경우 그대로, BottleNeck일 경우 4를 곱한다.
        self.linear = nn.Linear(512 * block.mul, num_classes)

    # 다양한 Architecture 생성을 위해 make_layer로 Sequential 생성
    def make_layer(self, block, out_planes, num_blocks, stride):
        # layer 앞부분에서만 크기를 절반으로 줄이므로, 아래와 같은 구조
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            layers.append(block(self.in_planes, out_planes, strides[i]))
            self.in_planes = block.mul * out_planes
        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.conv1(x)

        out = F.relu(out)

        out = self.maxpool1(out)

        out = self.layer1(out)

        out = self.layer2(out)

        out = self.layer3(out)

        out = self.layer4(out)

        out = self.avgpool(out)

        out = torch.flatten(out, 1)
        out = self.linear(out)

        return out

class ArcMarginProduct(nn.Module):
    def __init__(self, in_feature=128, out_feature=10575, s=32.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_feature, in_feature))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))

        #print(cosine.size())
        # cos(theta + m)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        #print(phi.size())

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        #print(one_hot)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        return output




class ganGenerator1(torch.nn.Module):

    def __init__(self,dNoise,dHidden):
        super(ganGenerator1, self).__init__()

        self.dNoise = dNoise
        self.dHidden = dHidden

        self.G = nn.Sequential(
            nn.Linear(self.dNoise, self.dHidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.dHidden,self.dHidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.dHidden, 28*28),
            nn.Tanh()
        )

    def forward(self, x):

        out = self.G(x)
        out = out.view(-1,1,28,28)

        return out


class ganDiscriminator1(torch.nn.Module):

    def __init__(self, dHidden):
        super(ganDiscriminator1, self).__init__()

        self.dHidden = dHidden

        self.D = nn.Sequential(
                nn.Linear(28*28, self.dHidden),
                nn.LeakyReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.dHidden, self.dHidden),
                nn.LeakyReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.dHidden, 1),
                nn.Sigmoid()
            )

    def forward(self, x):

        x = x.view(-1,28*28)
        out = self.D(x)

        return out

class MyBYOLMODEL(nn.Module):
    def __init__(self,modelKind,backboneOutFeature,LinNum,usePretrained=True):
        super(MyBYOLMODEL,self).__init__()

        # self.backboneOutFeature = backboneOutFeature
        # self.LinNum = LinNum
        # self.usePretrained = usePretrained

        self.backbone = models.resnet50(pretrained=True)

        # if modelKind == 'effnet-b0':
        #     print('loading effnet-b0')
        #     self.backbone = models.efficientnet_b0(pretrained=self.usePretrained,num_classes=self.backboneOutFeature)
        #     self.backbone.avgpool= GeM()
        #     self.backbone.classifier = nn.Identity()
        #
        # elif modelKind == 'effnet-b1':
        #     print('loading effnet-b1')
        #     self.backbone = models.efficientnet_b1(pretrained=self.usePretrained,num_classes=self.backboneOutFeature)
        #     self.backbone.avgpool = GeM()
        #     self.backbone.classifier = nn.Identity()
        #
        # elif modelKind == 'effnet-b2':
        #     print('loading effnet-b2')
        #     self.backbone = models.efficientnet_b2(pretrained=self.usePretrained,num_classes=self.backboneOutFeature)
        #     self.backbone.avgpool = GeM()
        #     self.backbone.classifier = nn.Identity()
        #
        # elif modelKind == 'effnet-b3':
        #     print('loading effnet-b3')
        #     self.backbone = models.efficientnet_b3(pretrained=self.usePretrained,num_classes=self.backboneOutFeature)
        #     self.backbone.avgpool = GeM()
        #     self.backbone.classifier = nn.Identity()
        #
        # elif modelKind == 'effnet-b4':
        #     print('loading effnet-b4')
        #     self.backbone = models.efficientnet_b4(pretrained=self.usePretrained,num_classes=self.backboneOutFeature)
        #     self.backbone.avgpool = GeM()
        #     self.backbone.classifier = nn.Identity()
        #
        # elif modelKind == 'effnet-b5':
        #     print('loading effnet-b5')
        #     self.backbone = models.efficientnet_b5(pretrained=self.usePretrained,num_classes=self.backboneOutFeature)
        #     self.backbone.avgpool = GeM()
        #     self.backbone.classifier = nn.Identity()
        #
        # elif modelKind == 'effnet-b6':
        #     print('loading effnet-b6')
        #     self.backbone = models.efficientnet_b6(pretrained=self.usePretrained,num_classes=self.backboneOutFeature)
        #     self.backbone.avgpool = GeM()
        #     self.backbone.classifier = nn.Identity()
        #
        # elif modelKind == 'effnet-b7':
        #     print('loading effnet-b7')
        #     self.backbone = models.efficientnet_b7(pretrained=self.usePretrained,num_classes=self.backboneOutFeature)
        #     self.backbone.avgpool = GeM()
        #     self.backbone.classifier = nn.Identity()
        #
        # elif modelKind == 'resnext-101':
        #     print('loading resnext 101')
        #     self.backbone = models.resnext101_32x8d(pretrained=self.usePretrained,num_classes=self.backboneOutFeature)
        #     self.backbone.avgpool = GeM()
        #     self.backbone.fc = nn.Identity()
        #
        # elif modelKind == 'densenet-161':
        #     print('loading densenet 161')
        #     self.backbone = models.densenet161(pretrained=self.usePretrained,num_classes=self.backboneOutFeature)
        #     self.backbone.classifier = nn.Identity()
        #
        # elif modelKind == 'densenet-201':
        #     print('loading densenet 201')
        #     self.backbone = models.densenet201(pretrained=self.usePretrained,num_classes=self.backboneOutFeature)
        # elif modelKind == 'regnet':
        #     print('loading regnet y 32gf')
        #     self.backbone = models.regnet_x_32gf(pretrained=self.usePretrained,num_classes= self.backboneOutFeature)
        # elif modelKind == 'effnet-v2s':
        #     print('loading efficientnet v2 small')
        #     self.backbone = effnetv2_s(num_classes=self.backboneOutFeature)
        # elif modelKind == 'effnet-v2m':
        #     print('loading efficientnet v2 middle')
        #     self.backbone = effnetv2_m(num_classes=self.backboneOutFeature)
        # elif modelKind == 'effnet-v2l':
        #     print('loading efficientnet v2 large')
        #     self.backbone = effnetv2_l(num_classes=self.backboneOutFeature)
        # elif modelKind == 'effnet-v2xl':
        #     print('loading efficientnet v2 extra large')
        #     self.backbone = effnetv2_xl(num_classes=self.backboneOutFeature)
        # elif modelKind == 'resnet-18':
        #     print('loading resnet 18 awef')
        #     self.backbone = models.resnet18(pretrained=self.usePretrained)
        # elif modelKind == 'resnet-50':
        #     print('loading resnet 50aawef')
        #     self.backbone = models.resnet50(pretrained=True)
        #     # self.backbone = models.resnet50(pretrained=self.usePretrained)
        # elif modelKind == 'resnet-101':
        #     print('loading resnet 101')
        #     self.backbone = models.resnet101(pretrained=self.usePretrained)
        # elif modelKind == 'resnet-152':
        #     print('loading resnet 152')
        #     self.backbone = models.resnet152(pretrained=self.usePretrained)
        # else:
        #     print(f'loading model {modelKind} from timm ....')
        #     self.backbone = timm.create_model(model_name=modelKind,pretrained=self.usePretrained)


        # testTensor = torch.randn((1,3,600,600))
        # self.inFeature = self.backbone(testTensor).size(1)

        # self.lin1 = nn.Linear(in_features=self.inFeature, out_features=self.LinNum)
        # self.lin1 = nn.Linear(in_features=self.backboneOutFeature, out_features=self.LinNum)


    def forward(self,x):

        out = self.backbone(x)
        # out = self.lin1(out)

        return out

class myCluster4SPICE(nn.Module):
    def __init__(self,
                 inputDim,
                 dim1,
                 clusters,
                 lossMethod='CE'):
        super(myCluster4SPICE, self).__init__()

        self.inputDim = inputDim
        self.dim1 = dim1
        self.clusters = clusters


        self.MLP = nn.Sequential(
            nn.Linear(in_features=self.inputDim,out_features=self.dim1),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.dim1,out_features=self.clusters)
        )

        self.lossMethod = lossMethod
        if self.lossMethod == 'CE':
            self.LOSS = nn.CrossEntropyLoss()

    def getLoss(self,pred,label):

        return self.LOSS(pred,label)

    def forward(self,x):

        out = self.MLP(x)

        return out


class myMultiCluster4SPICE(nn.Module):
    def __init__(self,
                 inputDim,
                 dim1,
                 dim2,
                 numHead,
                 lossMethod='CE'):
        super(myMultiCluster4SPICE, self).__init__()

        self.inputDim = inputDim
        self.dim1 = dim1
        self.dim2 = dim2
        self.numHead = numHead
        self.lossMethod = lossMethod

        for h in range(self.numHead):
            headH = myCluster4SPICE(inputDim=self.inputDim,
                                    dim1=self.dim1,
                                    dim2=self.dim2,
                                    lossMethod=self.lossMethod)
            self.__setattr__(f'eachHead_{h}',headH)


    def forward(self,x):

        totalForwardResult = []

        for h in range(self.numHead):
            eachForwardResult = self.__getattr__(f'eachHead_{h}').forward(x)
            totalForwardResult.append(eachForwardResult)

        return totalForwardResult


    def getTotalLoss(self,x,label):

        totalLoss = {}

        for h in range(self.numHead):
            eachLossH = self.__getattr__(f'eachHead_{h}').getLoss(pred=x,label=label)
            totalLoss[f'eachHead_{h}'] = eachLossH

        return totalLoss






























































