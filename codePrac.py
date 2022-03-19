import torch
import os
import shutil
from save_funcs import createDirectory
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.datasets import MNIST
from MK_NOISED_DATA import mk_noisy_data
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms

def imshow(img):
    img = (img+1)/2
    img = img.squeeze()
    np_img = img.numpy()
    plt.imshow(np_img, cmap='gray')
    plt.show()

def imshow_grid(img):
    img = utils.make_grid(img.cpu().detach())
    img = (img+1)/2
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()
d_noise  = 100
d_hidden = 256

def sample_z(batch_size = 1, d_noise=100):
    return torch.randn(batch_size, d_noise)

G = nn.Sequential(
    nn.Linear(d_noise, d_hidden),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(d_hidden,d_hidden),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(d_hidden, 28*28),
    nn.Tanh()
)

# 노이즈 생성하기
z = sample_z(batch_size=10)
# 가짜 이미지 생성하기
img_fake = G(z).view(-1,1,28,28)
print(img_fake.size())
# 이미지 출력하기
imshow_grid(img_fake.cpu().detach())

