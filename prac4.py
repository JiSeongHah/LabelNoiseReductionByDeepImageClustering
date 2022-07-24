# import os
# import pickle
# 
# basedir = '/home/a286winteriscoming/SCAN_imagenets/'
# dir = basedir+'imagenet50'
# lst = os.walk(dir)
# 
# totalPathLst = []
# for walk in lst:
#     eachPath = walk[0]
#     fles = walk[2]
# 
#     for fle in fles:
#         totalPathLst.append(os.path.join(eachPath,fle))


# with open(basedir+'imagenet50_PathLst.pkl','wb') as f:
#     pickle.dump(totalPathLst,f)

# with open(basedir+'imagenet200_LabelDict.pkl','rb') as F:
#     mylabelDict = pickle.load(F)

# with open(basedir+'imagenet50_PathLst.pkl','rb') as f:
#     myLst = pickle.load(f)
#
# for i in myLst:
#     print(i)
    # label = mylabelDict[i.split('/')[-2]]
    # print(label)


# baseDir = '/home/a286/hjs_dir1/mySCAN0/SCAN_imagenets/'
#
# import pickle
# with open(baseDir+'imagenet200_PathLst.pkl','rb') as F:
#     myDict = pickle.load(F)
#
# for i in myDict:
#     print(i)

























# with open(dir+'imagenet200.txt') as F:
#     lst = F.readlines()
#
#
# imagenet50labelDict= {}
# for idx,i in enumerate(lst):
#     imagenet50labelDict[i.split(' ')[0]] = idx
#
#
# # with open(dir+'imagenet200_LabelDict.pkl','wb') as FF:
# #     pickle.dump(imagenet50labelDict,FF)
#
# with open(dir+'imagenet200_LabelDict.pkl','rb') as f:
#     myDict= pickle.load(f)
#
# for k,v in myDict.items():
#     print(k,v)



#
# from SCAN_usefulUtils import saveTinyImagenetPathLstAndLabelDict
# dir = '/home/a286/hjs_dir1/mySCAN0/SCAN_imagenets/'
# do =  saveTinyImagenetPathLstAndLabelDict(dir)
# dir = '/home/a286/hjs_dir1/mySCAN0/SCAN_imagenets/'
# import pickle
# with open(dir+'tinyImagenet_PathLst.pkl','rb') as F:
#     myDict = pickle.load(F)
#
# for i in myDict:
#     print(i)


















































































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
#
import torch
from MY_MODELS import callAnyResnet
import torchvision.models as models
from collections import OrderedDict
#
# model = callAnyResnet(modelType = 'resnet50',
#                       numClass = 13 )
#
# # model = models.__dict__['resnet50']()
#
#
# model.fc = torch.nn.Identity()
#
#
# import torch
# from collections import OrderedDict
#
# dir = '/home/a286winteriscoming/Downloads/pretrainedModels/moco_v2_800ep_pretrain.pth.tar'
# # dir = '/home/a286winteriscoming/Downloads/pretrainedModels/moco_v1_200ep_pretrain.pth.tar'
# modelDict = torch.load(dir)
#
#
#
# newModelDict= dict()
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
#
#
# x = torch.randn(5,3,224,224)
#
# print(model(x).size())


# import pickle
# import os
#
# # lst = os.listdir('/home/a286/hjs_dir1/mySCAN0/')
# # for i in lst:
# #     print(i)
# with open('/home/a286/hjs_dir1/mySCAN0/SCAN_imagenets/imagenet10_LabelDict.pkl','rb') as F:
#     myDict= pickle.load(F)
#
# for i,v in myDict.items():
#     print(i,v)



from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms


dt = CIFAR10(root='~/',train=True,download=True,transform=transforms.ToTensor())

print(dt.__getitem__(31))


# ds = DataLoader(dt, batch_size=32,shuffle=True,num_workers=2)
#
# for idx,i in enumerate(ds):
#     print(i[1].size())
# for idx,i in enumerate(ds):
#     print(idx)






import torch
from SCAN_usefulUtils import Pseudo2Label

#
#
# x = torch.randint(0,5,(5,))
#
# print(x)
#
# mask = torch.randint(0,2,(5,)) ==1
# print(mask)
#
# print(x[mask])

x = torch.randint(0,2,(10,))
y = torch.randint(0,2,(10,))

print(torch.mean((x==y).float()))



# dic = {
#     0:0,
#     1:1,
#     2:2
# }
#
# inputs = torch.randint(0,3,(5,))
# print(inputs,111)
# labels = Pseudo2Label(dic,inputs)
# print(labels,222)
#
#
# x = torch.argmax(torch.randn(3,2))
# print(x.type())
#






































































