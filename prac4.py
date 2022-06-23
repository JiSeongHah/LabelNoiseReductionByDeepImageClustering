#
# import torch
# from torch import nn
# import torchvision
#
# from lightly.data import LightlyDataset
# from lightly.data import SwaVCollateFunction
# from lightly.loss import SwaVLoss
# from lightly.models.modules import SwaVProjectionHead
# from lightly.models.modules import SwaVPrototypes

#
# class SwaV(nn.Module):
#     def __init__(self, backbone):
#         super().__init__()
#         self.backbone = backbone
#         self.projection_head = SwaVProjectionHead(512, 512, 128)
#         self.prototypes = SwaVPrototypes(128, n_prototypes=512)
#
#     def forward(self, x):
#         x = self.backbone(x).flatten(start_dim=1)
#         x = self.projection_head(x)
#         x = nn.functional.normalize(x, dim=1, p=2)
#         p = self.prototypes(x)
#         return p
#
#
# resnet = torchvision.models.resnet18()
# backbone = nn.Sequential(*list(resnet.children())[:-1])
# model = SwaV(backbone)
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)
#
# # we ignore object detection annotations by setting target_transform to return 0
# pascal_voc = torchvision.datasets.VOCDetection(
#     "datasets/pascal_voc", download=True, target_transform=lambda t: 0
# )
# dataset = LightlyDataset.from_torch_dataset(pascal_voc)
# # or create a dataset from a folder containing images or videos:
# # dataset = LightlyDataset("path/to/folder")
#
# collate_fn = SwaVCollateFunction()
#
# dataloader = torch.utils.data.DataLoader(
#     dataset,
#     batch_size=128,
#     collate_fn=collate_fn,
#     shuffle=True,
#     drop_last=True,
#     num_workers=8,
# )
#
# criterion = SwaVLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
# print("Starting Training")
# for epoch in range(10):
#     total_loss = 0
#     for batch, _, _ in dataloader:
#         model.prototypes.normalize()
#         multi_crop_features = [model(x.to(device)) for x in batch]
#         high_resolution = multi_crop_features[:2]
#         low_resolution = multi_crop_features[2:]
#         loss = criterion(high_resolution, low_resolution)
#         total_loss += loss.detach()
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#     avg_loss = total_loss / len(dataloader)
#     print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")

import numpy as np

# label = np.array([i*i for i in range(10)])
#
# idx = np.array([[7,3],[5,2]])
#
# print(label)
# print(np.take(label,idx))

# x = np.array([-i for i in range(10)])
# print(x)
# print(x.shape)
#
# x = np.expand_dims(x,axis=1)
# print(x)
# print(x.shape)
#
# y = np.repeat(x,20,axis=1)
# print(y)
# print(y.shape)
# y = np.array([x for i in range(10)])
# print(y)
# print(y.T)
# print(np.equal(y,y.T))
# import torch
#
# x = np.array([i*i for i in range(10)])
# z = torch.from_numpy(x)
# print(z.size())
# print(z)
# zz = z.repeat([1,20])
# print(zz.size())
# print(zz)
# #
# import pickle
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# import torch.nn.functional as F
#
# clusterNum = 3
# embedSize = 5
# bsize = 7
# topkNum = 4
#
# class TEST():
#
#     classVar = 'juhyeon'
#
#     def __init__(self):
#
#         self.a = 1
#
#     # def __setattr__(self, key, value):
#     #     print(f' {key} done with value : {value}')
#     #     self.__dict__[key] = value
#
#
#
#
#     def doit(self):
#         for i in range(10):
#             self.__setattr__(f'head_{i}',i**i)
#
#
# test = TEST()
# print(TEST.classVar)
# test.doit()
# print(test.__dict__)
import torch

x= sum(i for i in range(10))
print(x)


#
# cosineSim = torch.randn(bsize,clusterNum)
# print(cosineSim)
# topkSim = torch.topk(cosineSim,dim=0,k=topkNum).indices
#
# print(topkSim)
#
# res = torch.zeros_like(cosineSim).scatter_(0,topkSim,1)
# NULL = (res ==0)*(-1e9)
# final = res
#
# test = torch.sum(res,dim=1)
# print(final)
# print(test)
# print(test != 0)
#
# final = final[test != 0]
# print(final)
#
#

# target = F.softmax(final,dim=1)
#
# pred = torch.randn(bsize,clusterNum)
#
# loss = torch.nn.CrossEntropyLoss()
#
# print(loss(pred,target))



# eachVecs = torch.randn(bsize,embedSize)
# print(eachVecs)
#
# probs = torch.randn(bsize,clusterNum)
# soft = torch.nn.Softmax(dim=1)
# softProbs = soft(probs)
# print(softProbs)
#
# topkConf = torch.topk(softProbs,dim=0,k=topkNum).indices
# print(topkConf)
#
# lst = []
# for eachCluster in range(clusterNum):
#     eachTopk = topkConf[:,eachCluster]
#     print(eachTopk)
#     selectedTensor = torch.index_select(eachVecs,dim=0,index=eachTopk)
#     print(selectedTensor)
#     # print(selectedTensor.size())
#     sumedTensor = torch.sum(selectedTensor,dim=0)
#     print(sumedTensor.size())
#     lst.append(sumedTensor)
#
#
# lst = torch.stack(lst)
#
# SUM = torch.sum(lst.mul(lst),dim=1)
# for i,each in zip(lst,SUM):
#     ver1 = torch.div(i,torch.sqrt(each))
#     ver2 = F.normalize(i.unsqueeze(0))
#     print(ver1)
#     print(ver2)
#
#
#






#
# baseDir = '/home/a286winteriscoming/'
# eachLst =['test16_500_result.pkl',
#        'test32_500_result.pkl',
#        'test64_800_result.pkl',
#        'test128_800_result.pkl',
#        'test2000_result.pkl']
#
# embLst = [16,32,64,128,256]
# nNeighLst = [10 * (i + 1) for i in range(10)]
# neighThresholdLst = [0.91, 0.93, 0.95, 0.97, 0.99]
#
# totalBaseline = []
# totalCompareLine = []
#
# for each,emb in zip(eachLst,embLst):
#     with open(baseDir + each, 'rb') as f:
#         myDict = pickle.load(f)
#
#     baselineLst = myDict['baselineAcc']
#     newAccLst = myDict['newAcc']
#
#     totalBaseline.append(np.mean(baselineLst))
#     totalCompareLine.append(np.mean(newAccLst))
#
# plt.plot(embLst,totalBaseline,'b')
# plt.plot(embLst,totalCompareLine,'r')
# plt.xlabel('embedding size')
# plt.ylabel('mean accuracy')
# plt.savefig('/home/a286winteriscoming/plotRESULT_Total.png')
# plt.close()
# plt.cla()
# plt.clf()

    # for idx,neighThreshold in enumerate(neighThresholdLst):
    #     baseline = baselineLst[idx::5]
    #     compare = newAccLst[idx::5]
        # plt.plot(nNeighLst,baseline,'b')
        # plt.plot(nNeighLst,compare,'r')
        # plt.xlabel('Number of nearest neighbor')
        # plt.ylabel('accuracy')
        # plt.title('embed size '+str(emb)+' threshold '+str(neighThreshold))
        # plt.savefig('/home/a286winteriscoming/plotRESULT_'+str(emb)+'_'+str(neighThreshold)+'.png')
        # plt.close()
        # plt.cla()
        # plt.clf()
        # print(f'saving plot for embed size : {emb} and threshold : {neighThreshold}')








