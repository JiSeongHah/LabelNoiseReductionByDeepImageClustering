import torch
import os
import shutil
from save_funcs import createDirectory
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.datasets import MNIST
from MK_NOISED_DATA import mk_noisy_data


# dir1= '/home/a286/hjs_dir1/dir5/'
# dir2=  '/home/a286/hjs_dir1/dir6/'
#
#
# DirLst = os.listdir(dir1)
#
# for eachDir in DirLst:
#     createDirectory(dir2+eachDir)
#     shutil.copy(dir1+eachDir+'/RL_reward_plot.png',dir2+eachDir+'/RL_reward_plot.png')
#     print(f'{dir1+eachDir+"/RL_reward_plot.png"} copy done')

# RL_train_dataset = MNIST('~/', train=True, download=True)
#
# RL_train_data = RL_train_dataset.data.numpy()
# RL_train_label = RL_train_dataset.targets.numpy()
#
# RL_train_label_zero = RL_train_label[RL_train_label == 0]
# RL_train_label_rest = RL_train_label[RL_train_label != 0]
#
# RL_train_data_zero = RL_train_data[RL_train_label == 0]
# RL_train_data_rest = RL_train_data[RL_train_label != 0]
#
#
#
# noise_ratio = 5
# split_ratio = int(5923*0.05)
#
#
# RL_train_data_zero_little = torch.from_numpy(mk_noisy_data(raw_data=RL_train_data_zero, noise_ratio=noise_ratio,
#                                       split_ratio=split_ratio, way='sum')).unsqueeze(1)
# RL_train_label_zero_little = torch.from_numpy(RL_train_label_zero[:])
#
#
# for i in range(30):
#     plt.imshow(np.squeeze(RL_train_data_zero_little[i+split_ratio,:,:],axis=0))
#     plt.savefig('/home/a286winteriscoming/dir7/'+str(i)+'.png')


x = torch.zeros(3)
y = torch.ones(5)
z = torch.zeros(4)


xxx = torch.cat((x,y),dim=0)

print(xxx.size())
print(xxx)

LowLst = [5*i/100 for i in range(16)]

print(LowLst)