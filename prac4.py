import torch
import numpy as np
import random
import time
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from MY_MODELS import myCluster4SPICE


x = torch.randn(10000,256)

model = myCluster4SPICE(inputDim=256,
                        dim1=256,
                        clusters=500)

# torch.nn.init.kaiming_normal(model.MLP)

y= model(x)
print(torch.argmax(y,dim=1))

lst = []
for eachunique in torch.unique(torch.argmax(y,dim=1)):
    print(f'{torch.count_nonzero((torch.argmax(y,dim=1)) == eachunique)} for {eachunique}')
    lst.append(torch.count_nonzero((torch.argmax(y,dim=1)) == eachunique))

import matplotlib.pyplot as plt

plt.plot(torch.unique(torch.argmax(y,dim=1)),lst)
plt.show()






# # x = torch.randn(5,)
# #
# # idx = torch.randint(0,4,(5,))
# #
# #
# # print(x)
# # print(idx)
# #
# # print(torch.index_select(x,0,idx))
# import torch.nn.functional as F
#
# x = torch.randn(7,2)
# idx= torch.randint(0,2,(7,))
#
# a = idx == 1
# b = idx == 3
#
# k = x[a]
# p = x[b]
#
# print(x)
# print(x[a])
# print(a)
# print(x[b])
# print(b)
# import torch.nn as nn
#
# class myCluster4SPICE(nn.Module):
#     def __init__(self,
#                  inputDim,
#                  dim1,
#                  dim2,
#                  lossMethod='CE'):
#         super(myCluster4SPICE, self).__init__()
#
#         self.inputDim = inputDim
#         self.dim1 = dim1
#         self.dim2 = dim2
#
#         self.MLP = nn.Sequential(
#             nn.Linear(in_features=self.inputDim,out_features=self.dim1),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_features=self.dim1,out_features=self.dim2)
#         )
#
#         self.lossMethod = lossMethod
#         if self.lossMethod == 'CE':
#             self.LOSS = nn.CrossEntropyLoss()
#
#     def getLoss(self,pred,label):
#
#         return self.LOSS(pred,label)
#
#     def forward(self,x):
#
#         out = self.MLP(x)
#
#         return out
#
#
# model =myCluster4SPICE(inputDim=2,dim1=16,dim2=13)
# print(p.type())
# print(model(k))
# print(model(p))
# print(model(p).size())


