import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch
import json
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW, Adam
import matplotlib.pyplot as plt
import copy
import pandas as pd
import numpy as np
import random
import time
import datetime
import os
import argparse
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import multiprocessing
from glob import glob
from torch.distributions import Categorical
from torchvision import models
from torchvision.datasets import MNIST
import csv
from CALAVG import cal_avg_error
from MY_MODELS import ganGenerator1, ganDiscriminator1,ResNet4one,BasicBlock4one
from save_funcs import mk_name,lst2csv,createDirectory
from REINFORCE_DATAMODULES import datamodule_4REINFORCE1
from REINFORCE_INNER_MODELS import Prediction_lit_4REINFORCE1
from MK_NOISED_DATA import mk_noisy_data
from save_funcs import load_my_model


class REINFORCE_GAN_TORCH(nn.Module):
    def __init__(self,
                 gamma,
                 eps,
                 dNoise,
                 dHidden,
                 rl_lr,
                 rl_b_size,
                 gan_trn_bSize,
                 gan_val_bSize,
                 reward_normalize,
                 val_data,
                 val_label,
                 rwd_spread,
                 beta4f1,
                 rl_stop_threshold,
                 test_fle_down_path,
                 model_save_load_path,
                 max_step_trn,
                 max_step_val,

                 reward_method,
                 ):
        super(REINFORCE_GAN_TORCH, self).__init__()

        self.test_fle_down_path = test_fle_down_path
        self.model_save_load_path = model_save_load_path

        self.dNoise = dNoise
        self.dHidden = dHidden

        ####################################MODEL SETTINGG##############################3

        self.REINFORCE_GAN_G = ganGenerator1(dNoise=self.dNoise,dHidden=self.dHidden)
        self.REINFORCE_GAN_G_BASE = copy.deepcopy(self.REINFORCE_GAN_G)
        self.REINFORCE_GAN_D = ganDiscriminator1(dHidden=self.dHidden)
        self.REINFORCE_DVRL = ResNet4one(block=BasicBlock4one, num_blocks=[2, 2, 2, 2], num_classes=2, mnst_ver=True)
        self.model_num_now = 0
        ####################################MODEL SETTINGG##############################3

        ##########################VARS for RL model##################################
        self.loss_lst_trn = []
        self.loss_lst_val = []

        self.policy_saved_log_probs_lst = []
        self.R_lst = []
        self.policy_saved_log_probs_lst_val = []
        self.R_lst_val = []

        self.total_reward_lst_trn = []

        self.automatic_optimization = False
        self.gamma = gamma
        self.eps = eps
        self.rl_lr = rl_lr
        self.rl_b_size = rl_b_size
        self.rl_stop_threshold = rl_stop_threshold
        self.reward_normalize = reward_normalize
        self.max_step_trn = max_step_trn
        self.max_step_val = max_step_val
        self.beta4f1 = beta4f1

        self.reward_method = reward_method

        self.pi4window = 0

        self.gan_trn_bSize = gan_trn_bSize
        self.gan_val_bSize = gan_val_bSize


        self.lossLstLossG = []
        self.lossLstLossD = []

        self.ProbFakeLst = []
        self.ProbRealLst = []


        ########################do validation dataset change##########################
        self.val_label_zero = val_label[val_label == 0]
        self.val_label_rest = val_label[val_label != 0]

        self.val_data_zero = val_data[val_label == 0]
        self.val_data_rest = val_data[val_label != 0]
        self.val_label_rest = torch.ones_like(self.val_label_rest)

        self.val_data = torch.cat((self.val_data_rest,self.val_data_zero),dim=0)
        self.val_label = torch.cat((self.val_label_rest,self.val_label_zero),dim=0)

        del self.val_label_zero
        del self.val_label_rest
        del self.val_data_rest
        del self.val_data_zero
        ######################################do validation dataset change#################
        self.rwd_spread = rwd_spread

        self.optimizerD = Adam(self.REINFORCE_GAN_D.parameters(),
                              lr=self.rl_lr,  # 학습률
                              eps=self.eps  # 0으로 나누는 것을 방지하기 위한 epsilon 값
                              )

        self.optimizerG = Adam(self.REINFORCE_GAN_G.parameters(),
                               lr=self.rl_lr,  # 학습률
                               eps=self.eps  # 0으로 나누는 것을 방지하기 위한 epsilon 값
                               )

        ##########################VARS for RL model##################################




        test_dataset = MNIST(self.test_fle_down_path, train=False, download=True)

    def forward(self, x):

        GeneratedImages = self.REINFORCE_GAN_G(x)

        return GeneratedImages

    def get_gaussianNoise_z(self,BSize):

        return torch.randn(BSize,self.dNoise)

    def flush_lst(self):
        self.policy_saved_log_probs_lst = []
        self.policy_saved_log_probs_lst_val = []
        self.R_lst = []
        self.R_lst_val = []
        print('flushing lst on pl level complete')

    def imshow(self,img,saveDir):
        img = (img + 1) / 2
        img = img.squeeze()
        np_img = img.numpy()
        plt.imshow(np_img, cmap='gray')
        # plt.show()
        plt.savefig(saveDir + 'GENERATED_IMAGE.png', dpi=200)
        plt.close()

    def imshow_grid(self,img,saveDir,showNum):
        img = utils.make_grid(img.cpu().detach())
        img = (img + 1) / 2
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

        randNum = random.random()
        if randNum < 0.1 and showNum % 10 ==0:
            plt.show()
        plt.savefig(saveDir + 'GENERATED_IMAGE_GRID.png', dpi=200)
        plt.close()

    def REINFORCE_LOSS(self, action_prob, reward):

        return action_prob * reward

    def GAN_LOSS(self,GorD,p_fake,p_real=None):

        if GorD == 'D':
            lossFake = -1 * torch.log(1.0 - p_fake)
            lossReal = -1 * torch.log(p_real)

            lossD = (lossReal+lossFake).mean()

            return lossD
        if GorD == 'G':
            lossG = -1 * torch.log(p_fake).mean()

            return lossG

    def training_step(self, td2gen, tl2gen):

        self.REINFORCE_GAN_G.train()
        self.REINFORCE_GAN_D.train()
        self.REINFORCE_GAN_G_BASE.train()

        train_data = TensorDataset(td2gen, tl2gen)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data,
                                      sampler=train_sampler,
                                      batch_size=self.gan_trn_bSize,
                                      num_workers=1)

        for idx,(bInput, bLabel) in enumerate(train_dataloader):

            bInput = bInput.float()

            self.optimizerG.zero_grad()
            self.optimizerD.zero_grad()

            noiseZ = self.get_gaussianNoise_z(bInput.size(0))

            p_real = self.REINFORCE_GAN_D(bInput)
            p_fake = self.REINFORCE_GAN_D(self.REINFORCE_GAN_G(noiseZ))

            lossD = self.GAN_LOSS(p_fake=p_fake,p_real=p_real,GorD='D')
            lossD.backward()
            self.optimizerD.step()

            noiseZ = self.get_gaussianNoise_z(bInput.size(0))
            p_fake = self.REINFORCE_GAN_D(self.REINFORCE_GAN_G(noiseZ))

            self.optimizerG.zero_grad()
            lossG = self.GAN_LOSS(p_fake=p_fake,GorD='G')
            lossG.backward()
            self.optimizerG.step()

        self.REINFORCE_GAN_G.eval()
        self.REINFORCE_GAN_D.eval()
        self.REINFORCE_GAN_G_BASE.eval()

    def validation_step(self, vd2gen,vl2gen):

        self.REINFORCE_GAN_G.eval()
        self.REINFORCE_GAN_D.eval()
        self.REINFORCE_GAN_G_BASE.eval()

        val_data = TensorDataset(vd2gen, vl2gen)
        validation_dataloader = DataLoader(val_data,
                                      shuffle=False,
                                      batch_size=self.gan_val_bSize,
                                      num_workers=1)

        for idx, (bInput, bLabel) in enumerate(validation_dataloader):
            bInput = bInput.float()
            self.optimizerG.zero_grad()
            self.optimizerD.zero_grad()

            noiseZ = self.get_gaussianNoise_z(bInput.size(0))

            p_real = self.REINFORCE_GAN_D(bInput)
            p_fake = self.REINFORCE_GAN_D(self.REINFORCE_GAN_G(noiseZ))

            lossD = self.GAN_LOSS(p_fake=p_fake, p_real=p_real, GorD='D')
            lossG = self.GAN_LOSS(p_fake=p_fake,GorD='G')

            self.lossLstLossD.append(lossD.item())
            self.lossLstLossG.append(lossG.item())
            self.ProbRealLst.append(torch.mean(p_real).item())
            self.ProbFakeLst.append(torch.mean(p_fake).item())
            print(f'{idx}/{len(validation_dataloader)} done with'
                  f'lossD : {lossD} and lossG : {lossG}'
                  f'p_real: {torch.mean(p_real).item()} and p_fake: {torch.mean(p_fake).item()}')


            self.imshow_grid(self.REINFORCE_GAN_G(noiseZ).cpu().clone().detach().view(-1, 1, 28, 28),
                             saveDir=self.model_save_load_path,
                             showNum=idx
                             )


        self.REINFORCE_GAN_G.train()
        self.REINFORCE_GAN_D.train()
        self.REINFORCE_GAN_G_BASE.train()


    def STARTTRNANDVAL(self,data,label):

        print('start training step....')
        self.training_step(td2gen=data,tl2gen=label)
        print('training step complete!!!')
        print('validation step start...')
        self.validation_step(vd2gen=data,vl2gen=label)
        print('validation step complete!!!')

        fig = plt.figure()
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(range(len(self.lossLstLossD)), self.lossLstLossD)
        ax1.set_title('loss D')
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(range(len(self.lossLstLossG)), self.lossLstLossG)
        ax2.set_title('loss G')
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(range(len(self.ProbRealLst)), self.ProbRealLst)
        ax3.set_title('Prob D')
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(range(len(self.ProbFakeLst)), self.ProbFakeLst)
        ax4.set_title('Prob G')

        print(f'self.test_fle_down_path is : {self.test_fle_down_path}testplot.png')
        plt.savefig(self.test_fle_down_path + 'GAN_RESULT_PLOT.png', dpi=200)
        print('saving plot complete!')
        plt.close()


















