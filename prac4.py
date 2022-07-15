import os


# def delFileOrFolders(dir,exceptionLst):
#
#     fileLst = os.listdir(dir)
#
#     for eachFile in fileLst:
#         if eachFile not in exceptionLst:
#             os.remove(dir+eachFile)
#             print(f'{dir}+{eachFile} removed')
#
#
#
# dir = '/home/a286/hjs_dir1/mySCAN0/dirResultSTL10_2'
#
# lst = os.walk(dir)
# totalLst = []
# for walks in lst:
#     dirs = walks[0]
#     fles = walks[2]
#     for fle in fles:
#         if fle.endswith('.pt'):
#             totalLst.append(os.path.join(dirs,fle))
#
# for i in totalLst:
#     num = float(i.split('/')[-1].split('.')[0])
#
#     if num >= 120:
#         os.remove(i)
#         print(f'{i} removed')
#



#
# import torch
# from MY_MODELS import callAnyResnet
# import torchvision.models as models
# from collections import OrderedDict
#
# model = callAnyResnet(modelType = 'resnet50',
#                       numClass = 13 )
#
# # model = models.__dict__['resnet50']()
#
#
#
#
#
# dir = '/home/a286winteriscoming/Downloads/pretrainedModels/moco_v2_800ep_pretrain.pth.tar'
# # dir = '/home/a286winteriscoming/Downloads/pretrainedModels/moco_v1_200ep_pretrain.pth.tar'
# modelDict = torch.load(dir)
# newModelDict= OrderedDict()
#
#
# for k,v in modelDict['state_dict'].items():
#     if k[:17] == 'module.encoder_q.':
#         name = 'backbone.'+k[17:]
#     else:
#         name = k
#     newModelDict[name] = v
#
#
# for modelLayer,dictLayer in zip(model.state_dict(),newModelDict):
#     print(modelLayer, '   ',dictLayer)
#
# missing = model.load_state_dict(newModelDict,strict=False)
# print(missing)



import torch
from torchvision.datasets import CIFAR100

dt = CIFAR100(root='~/',train=True,download=True)

print(dt.targets)