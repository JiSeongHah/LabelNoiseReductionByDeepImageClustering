import torch
import numpy as np
import random
import time
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
# labelLen = 50000
#
# x = torch.randint(10,(labelLen,))
#
# print(x)
#
# y = torch.unique(x)
# print(torch.max(y),torch.min(y))
#
# z = torch.randint(10,(1,))
# print(z)

x = torch.randint(0,5,(10,))
y = torch.randint(0,2,(10,))

idx = y ==10
print(x)
print(x.size())
print(idx)
print(x[idx])
print(x[idx].size(0))
# print(torch.mode(x[idx]))








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


