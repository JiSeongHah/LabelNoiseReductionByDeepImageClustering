# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import math
#
# # output = torch.tensor([float(0.01*i) for i in range(12)]).view(1,3,2,2)
# #
# # loss = nn.CrossEntropyLoss()
# # target = torch.ones((1,2,2),dtype=torch.long)
# #
# # print(output)
# # print(output.size())
# # print(F.softmax(output,dim=1))
# #
# #
# # res = loss(output,target)
# # print(res)
# #
# # def sigmoid(x):
# #     return 1 / (1 + np.exp(-x))
# #
# #
# # res2 = [-math.log(0.3332) for i in range(4)]
# #
# # print(np.mean(res2))
# # import torch
# # import torch.nn.functional as F
# #
# # output = torch.tensor([i for i in range(24)]).view(4,3,2)
# # print(output.size())
# # print(output.reshape(-1,1))
# #
#
# #
# # w_lst = []
# # for i in range(25):
# #     if i ==0:
# #         w_lst.append(0)
# #     else:
# #         w_lst.append(1/24)
# # w_tensor = torch.tensor(w_lst)
# # print(w_tensor)
# #
# # loss = nn.CrossEntropyLoss(weight=w_tensor)
# #
# # output = torch.randn(1,25,128,128)
# # target = torch.tensor(torch.ones(1,25,128,128),dtype=torch.long)
# #
# # print((output==target))
#
#
# #
# # output = F.softmax(torch.randn((50,100,2,2))*0.001,dim=1)
# # target = torch.ones((50,100,2,2))
# # print(output)
# # mask_index = torch.sum(output, dim=(-1))
# # print(mask_index.size())
# # mask_index = torch.sum(mask_index, dim=(-1))
# # print(mask_index.size())
# # mask_index = mask_index > 0
# # print(mask_index)
# # mask_index[:,0] = False
# # print(mask_index)
# # print(output[mask_index,:,:].size())
# #
# # loss = nn.CrossEntropyLoss()
# # kkk = torch.tensor(torch.ones((50,2,2)),dtype=torch.long)
# # print(kkk.type())
# # print(loss(output[mask_index,:,:],target[mask_index,:,:]))
# # apap = mask_index.size()
#
#
# import tifffile
# from skimage import io
#
# dir = '/home/emeraldsword1423/result133/2021311627.csv'
# import csv
#
# with open(dir,'r') as f:
#     reader = csv.reader()
#     lines = reader.lin
#
#
#
#
#
# import csv
# 
# dir1 = '/home/emeraldsword1423/final_final_submission/FOCALLOSS_WEIGHTVER_w0.05_test123123test123123model_nameunet_n_blocks6_ngf32_.csv'
# dir2 = '/home/emeraldsword1423/final_final_submission/2021311627.csv'
# 
# with open(dir1,'r') as f:
#     lst1 = f.readlines()
# 
# with open(dir2,'r') as f:
#     lst2 = f.readlines()
# 
# 
# yes_count = 0
# no_count = 0
# 
# for i in range(len(lst1)):
#     if lst1[i] ==lst2[i]:
#         yes_count+=1
#         print('yes')
#     else:
#         no_count += 1
#         print('no')
# 
# print(yes_count,no_count)

# import torch
# x = torch.randn(5,3,2)
# print(x.size())
# print(x[3:].size())

import os
from save_funcs import createDirectory
import shutil

rootPath = '/home/a286/hjs_dir1/dirRL/'
createDirectory('/home/a286/hjs_dir1/dirRLBackup2/')
backupDir = '/home/a286/hjs_dir1/dirRLBackup2/'
eachFolderLst = list(map(lambda x:x+'/',os.listdir(rootPath)))

for eachFolder in eachFolderLst:
    eachFile = [file for file in os.listdir(rootPath+eachFolder) if file.endswith('.png')]

    for each in eachFile:
        createDirectory(backupDir+eachFolder)
        shutil.copy(rootPath+eachFolder+each,backupDir+eachFolder+each)
        print('done')
#

# notRemoveLst = [str(10000*i)+'.pth' for i in range(1000)] + ['Result.png','sample_submission.csv']
# 
# for eachFolder in eachFolderLst:
#     fullEachFolder = rootPath+eachFolder+'/'
#     eachFileLst = os.listdir(fullEachFolder)
#     if len(eachFileLst) == 0:
#         os.rmdir(fullEachFolder)
#         print(f'{fullEachFolder} removed because it is empty')
#     for eachFile in eachFileLst:
#         fullEachFilePath = fullEachFolder + eachFile
#         if eachFile not in notRemoveLst:
#             os.remove(fullEachFilePath)
#             print(f'{fullEachFilePath} removed')