import torch
import numpy as np
import random
import time
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader,TensorDataset
from MY_MODELS import callAnyResnet
import torch.nn.functional as F
import numpy as np


x = torch.randn(5,2)
y = torch.randint(0,5,(5,))

dt = TensorDataset(x,y)

print(x)
print(y)
print('-------------------')
for a,b in dt:
    print(a,b)