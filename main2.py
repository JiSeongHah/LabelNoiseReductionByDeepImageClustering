import torch
import numpy as np
import math
from PIL import Image, TiffImagePlugin
from torchvision import transforms
from libtiff import TIFF
from glob import glob
import tifffile as tiff
import time
from MODELS import ResnetGenerator
import torch.nn as nn

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def bce_loss(x,y):
    if y ==1:
        return -y*math.log(sigmoid(x))
    elif y ==0:
        return -(1-y)*math.log(1-sigmoid(x))



img_file_dir = '/media/emeraldsword1423/My Passport/project2_dataset/Training/[라벨]train-label/Zone-A-001/Zone-A-001/Zone-A-001_000_001_gt.tif'
#f_name = 'Zone-A-001_000_001.tif'
# f_lst = glob(img_file_dir)
# print(f_lst)
# img = TIFF.open(img_file_dir)

transformer = transforms.Compose([transforms.ToTensor()])

img = tiff.imread(img_file_dir)
img = np.asarray(img)
#img = Image.open(img_file_dir)
# print(img.shape)
# print(type(img))
# for i in range(128):
#     for j in range(128):
#         print(img[i,j])
#         time.sleep(0.3)
target = transformer(img).unsqueeze(1)



# print(target.size())


# input = torch.randn((1,3,128,128))
# model = ResnetGenerator(input_nc=3,output_nc=10)
# output = model(input)
# print(output.size())

target = torch.zeros((1)).unsqueeze(0)
output = torch.tensor([0.1,0.2,0.3]).unsqueeze(0)


print(target.size(),output.size())
loss = nn.MultiLabelSoftMarginLoss(reduction='mean')


msml  = loss(output,target)
print(msml)

bce = [bce_loss(output[0,0],0),bce_loss(output[0,1],0),bce_loss(output[0,2],0)]
print(np.mean(bce))


def label_shape_chage(label_tensor,label_num):
    tensor_shape_changed = torch.zeros((label_tensor.size(0),label_num,label_tensor.size(2),label_tensor.size(3)))

    tensor_shape_changed[range(label_tensor.size(0)), ]

    return tensor_shape_changed

import torch.nn.functional as F

x = torch.tensor([i for i in range(50)]).view(5,10)
print(x.size())

y = F.one_hot(x)
print(y.size())






